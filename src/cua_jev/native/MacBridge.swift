// Native AX/CGEvent and single-window ScreenCaptureKit transport.
// Only this caller-selected window is retained. No shell or AppleScript execution.
import AppKit
import ApplicationServices
import ScreenCaptureKit
import Foundation

struct BridgeError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
func fail(_ text: String) -> BridgeError { BridgeError(message: text) }
func attr(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
}
func string(_ element: AXUIElement, _ name: String) -> String {
    (attr(element, name) as? String) ?? ""
}
func elements(_ element: AXUIElement, _ name: String) -> [AXUIElement] {
    (attr(element, name) as? [AXUIElement]) ?? []
}
func bounds(_ element: AXUIElement) -> CGRect? {
    guard let p = attr(element, kAXPositionAttribute), let s = attr(element, kAXSizeAttribute),
          CFGetTypeID(p) == AXValueGetTypeID(), CFGetTypeID(s) == AXValueGetTypeID() else { return nil }
    var point = CGPoint.zero, size = CGSize.zero
    guard AXValueGetValue(p as! AXValue, .cgPoint, &point),
          AXValueGetValue(s as! AXValue, .cgSize, &size),
          [point.x, point.y, size.width, size.height].allSatisfy({ $0.isFinite }),
          size.width > 0, size.height > 0 else { return nil }
    return CGRect(origin: point, size: size)
}
func rectArray(_ rect: CGRect) -> [Double] {
    [rect.minX, rect.minY, rect.width, rect.height].map { Double($0) }
}
func jsonString(_ object: Any) throws -> String {
    String(data: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), encoding: .utf8)!
}
func screenLockStatus(_ session: [String: Any]?) -> Bool? {
    // The public session API can return nil outside a Quartz GUI session.
    // This observed lock key has no public SDK constant; absence or an
    // unexpected type is unknown, never evidence that the session is unlocked.
    guard let value = session?["CGSSessionScreenIsLocked"] as? NSNumber,
          CFGetTypeID(value) == CFBooleanGetTypeID() else { return nil }
    return value.boolValue
}
func currentScreenLockStatus() -> Bool? {
    screenLockStatus(CGSessionCopyCurrentDictionary() as? [String: Any])
}
func requireUnlockedSession(_ locked: Bool? = currentScreenLockStatus()) throws {
    if locked == true {
        throw fail("macOS screen is locked. Unlock the current session before selecting a window or starting a recording; CUA-JEV does not unlock it.")
    }
}
func requireAX() throws {
    guard AXIsProcessTrusted() else {
        throw fail("macOS Accessibility permission is required for the process launching CUA-JEV. Grant it in System Settings > Privacy & Security > Accessibility, then restart the task.")
    }
}
func requireCapture() throws {
    guard CGPreflightScreenCaptureAccess() else {
        throw fail("macOS Screen Recording permission is required for selected-window capture. Grant it in System Settings > Privacy & Security > Screen & System Audio Recording, then restart the task.")
    }
}
func windowInfos() -> [[String: Any]] {
    CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
}
func infoBounds(_ info: [String: Any]) -> CGRect? {
    guard let dict = info[kCGWindowBounds as String] as? NSDictionary else { return nil }
    return CGRect(dictionaryRepresentation: dict)
}

// Dock exposes a transparent, display-sized interaction plane on recent macOS.
// Its bounding rectangle is not evidence that the target pixels are occluded.
// Match the genuine system component narrowly; foreground and AX hit identity
// are still required after this CG-window prefilter.
func isDockInteractionPlane(bundleID: String?, bundlePath: String?, name: String?,
                            layer: Int?, rect: CGRect?, displayBounds: [CGRect]) -> Bool {
    guard bundleID == "com.apple.dock", bundlePath == "/System/Library/CoreServices/Dock.app",
          name == "Dock", layer == 20, let rect else { return false }
    return displayBounds.contains { $0 == rect }
}

func isDockInteractionPlane(_ info: [String: Any], at point: CGPoint) -> Bool {
    guard let pid = info[kCGWindowOwnerPID as String] as? Int32,
          let application = NSRunningApplication(processIdentifier: pid) else { return false }
    var displays = [CGDirectDisplayID](repeating: 0, count: 16)
    var count: UInt32 = 0
    guard CGGetDisplaysWithPoint(point, 16, &displays, &count) == .success else { return false }
    return isDockInteractionPlane(
        bundleID: application.bundleIdentifier,
        bundlePath: application.bundleURL?.standardizedFileURL.path,
        name: info[kCGWindowName as String] as? String,
        layer: info[kCGWindowLayer as String] as? Int,
        rect: infoBounds(info), displayBounds: displays.prefix(Int(count)).map(CGDisplayBounds)
    )
}

struct RecordingIssue: Error, LocalizedError, Equatable {
    let stage: String
    let domain: String
    let code: Int
    let message: String
    init(_ error: Error, stage: String) {
        if let issue = error as? RecordingIssue { self = issue; return }
        let native = error as NSError
        self.stage = stage; domain = native.domain; code = native.code
        message = native.localizedDescription
    }
    static func timeout(_ stage: String) -> RecordingIssue {
        RecordingIssue(NSError(domain: "CUAJEVRecordingTimeout", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Recording phase exceeded its deadline"]), stage: stage)
    }
    var errorDescription: String? { "\(stage): \(domain) (\(code)): \(message)" }
    var json: [String: Any] { ["stage": stage, "domain": domain, "code": code, "message": message] }
}

// ScreenCaptureKit invokes delegates on its own queues. Never read these flags
// directly from the main actor without the same lock used by callbacks.
final class RecordingState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var finished = false
    private var issues: [RecordingIssue] = []
    private var events: [[String: Any]] = []
    func mark(_ event: String, error: Error? = nil) {
        lock.lock(); defer { lock.unlock() }
        if event == "started" { started = true }
        if event == "finished" { finished = true }
        var entry: [String: Any] = ["event": event, "at": Date().timeIntervalSince1970]
        if let error {
            let issue = RecordingIssue(error, stage: event)
            issues.append(issue); entry["error"] = issue.json
        }
        if events.count < 64 { events.append(entry) }
    }
    func snapshot() -> (started: Bool, finished: Bool, issues: [RecordingIssue], events: [[String: Any]]) {
        lock.lock(); defer { lock.unlock() }
        return (started, finished, issues, events)
    }
}

final class CaptureCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Void, Error>?
    func complete(_ error: Error?) {
        lock.lock(); defer { lock.unlock() }
        if result == nil { result = error.map { .failure($0) } ?? .success(()) }
    }
    func snapshot() -> Result<Void, Error>? {
        lock.lock(); defer { lock.unlock() }; return result
    }
}

// Poll a callback result, not an async framework task: an uncancellable
// stopCapture child in a task group can outlive the timeout and hang its parent.
@MainActor
func boundedCapture(_ stage: String, timeout: TimeInterval = 5,
                    operation: (@escaping @Sendable (Error?) -> Void) -> Void) async throws {
    let completion = CaptureCompletion()
    let deadline = ProcessInfo.processInfo.systemUptime + timeout
    operation { completion.complete($0) }
    while true {
        if let result = completion.snapshot() { try result.get(); return }
        guard ProcessInfo.processInfo.systemUptime < deadline else { throw RecordingIssue.timeout(stage) }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
}

@MainActor
func waitForRecording(_ state: RecordingState, finished: Bool, timeout: TimeInterval) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + timeout
    while true {
        let snapshot = state.snapshot()
        if let issue = snapshot.issues.first { throw issue }
        if finished ? snapshot.finished : snapshot.started { return }
        guard ProcessInfo.processInfo.systemUptime < deadline else {
            throw RecordingIssue.timeout(finished ? "finalize_output" : "start_output")
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
}

struct RecordingStopResult {
    let captureStopped: Bool
    let issues: [RecordingIssue]
    let events: [[String: Any]]
}

struct RecordingFailure: Error, LocalizedError {
    let result: RecordingStopResult
    var errorDescription: String? { result.issues.compactMap(\.errorDescription).joined(separator: "; ") }
    var json: [String: Any] {
        ["capture_stopped": result.captureStopped, "issues": result.issues.map(\.json), "events": result.events]
    }
}

@MainActor
func finishRecording(_ state: RecordingState, finalizeTimeout: TimeInterval = 10,
                     stopTimeout: TimeInterval = 5, remove: () throws -> Void,
                     stop: (@escaping @Sendable (Error?) -> Void) -> Void) async -> RecordingStopResult {
    var issues: [RecordingIssue] = []
    var stopped = false
    do {
        state.mark("remove_output_requested")
        try remove()
        state.mark("output_removed")
        try await waitForRecording(state, finished: true, timeout: finalizeTimeout)
    } catch { issues.append(RecordingIssue(error, stage: "finalize_output")) }
    // Always stop the capture, even after writer failure; keep both errors.
    do {
        state.mark("stop_capture_requested")
        try await boundedCapture("stop_capture", timeout: stopTimeout, operation: stop)
        stopped = true; state.mark("capture_stopped")
    } catch { issues.append(RecordingIssue(error, stage: "stop_capture")) }
    let snapshot = state.snapshot()
    for issue in snapshot.issues where !issues.contains(issue) { issues.append(issue) }
    return RecordingStopResult(captureStopped: stopped, issues: issues, events: snapshot.events)
}

@available(macOS 15.0, *)
final class RecordingDelegate: NSObject, SCRecordingOutputDelegate, SCStreamDelegate {
    let state = RecordingState()
    func recordingOutputDidStartRecording(_ recordingOutput: SCRecordingOutput) { state.mark("started") }
    func recordingOutputDidFinishRecording(_ recordingOutput: SCRecordingOutput) { state.mark("finished") }
    func recordingOutput(_ recordingOutput: SCRecordingOutput, didFailWithError error: Error) {
        state.mark("recording_failed", error: error)
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) { state.mark("stream_failed", error: error) }
}

@MainActor
final class Bridge {
    var window: AXUIElement?
    var app: NSRunningApplication?
    var windowID: CGWindowID = 0
    var windowToken = ""
    var refs: [(String, AXUIElement)] = []
    var nextRef = 0
    var previousState: String?
    var stream: SCStream?
    var recordingDelegate: AnyObject?
    var recordingOutput: AnyObject?
    var recordingOutputAttached = false
    let captureBackground = CGColor(gray: 0.96, alpha: 1)
    let allowedRoles: [String: String] = [
        "AXButton": "Button", "AXCheckBox": "CheckBox", "AXRadioButton": "RadioButton",
        "AXPopUpButton": "ComboBox", "AXComboBox": "ComboBox", "AXTextField": "Edit", "AXTextArea": "Edit"
    ]

    func refFor(_ element: AXUIElement) -> String {
        if let pair = refs.first(where: { CFEqual($0.1, element) }) { return pair.0 }
        let ref = "c\(nextRef)"; nextRef += 1
        refs.append((ref, element))
        return ref
    }

    func select(_ request: [String: Any]) throws -> [String: Any] {
        guard stream == nil else { throw fail("Stop the current recording before selecting another window") }
        // A failed selection must not leave the previous target available.
        window = nil; app = nil; windowID = 0; windowToken = ""
        refs = []; nextRef = 0; previousState = nil
        try requireUnlockedSession()
        try requireAX()
        guard let pattern = request["title_pattern"] as? String, !pattern.isEmpty, pattern.count <= 240 else {
            throw fail("A bounded window title pattern is required")
        }
        let regex = try NSRegularExpression(pattern: pattern)
        let bundle = request["bundle_id"] as? String
        var matches: [(NSRunningApplication, AXUIElement, CGWindowID)] = []
        let infos = windowInfos()
        for application in NSWorkspace.shared.runningApplications where !application.isTerminated {
            if let bundle, !bundle.isEmpty, application.bundleIdentifier != bundle { continue }
            if application.activationPolicy != .regular { continue }
            let ax = AXUIElementCreateApplication(application.processIdentifier)
            AXUIElementSetMessagingTimeout(ax, 1.0)
            for candidate in elements(ax, kAXWindowsAttribute) {
                let title = string(candidate, kAXTitleAttribute)
                guard regex.firstMatch(in: title, range: NSRange(title.startIndex..., in: title)) != nil,
                      (attr(candidate, kAXMinimizedAttribute) as? Bool) != true,
                      let rect = bounds(candidate) else { continue }
                let found = infos.filter { info in
                    (info[kCGWindowOwnerPID as String] as? Int32) == application.processIdentifier &&
                    (info[kCGWindowLayer as String] as? Int) == 0 &&
                    infoBounds(info).map { abs($0.minX - rect.minX) < 2 && abs($0.minY - rect.minY) < 2 &&
                        abs($0.width - rect.width) < 2 && abs($0.height - rect.height) < 2 } == true
                }
                if found.count == 1, let id = found[0][kCGWindowNumber as String] as? UInt32 {
                    matches.append((application, candidate, id))
                }
            }
        }
        guard matches.count == 1 else {
            throw fail("Window selection must match exactly one visible AX window; found \(matches.count)")
        }
        (app, window, windowID) = matches[0]
        windowToken = UUID().uuidString; refs = []; nextRef = 0
        try focus()
        return try snapshot()
    }

    func selected() throws -> (AXUIElement, NSRunningApplication, CGRect) {
        try requireUnlockedSession()
        try requireAX()
        guard let window, let app, !app.isTerminated, let rect = bounds(window),
              (attr(window, kAXMinimizedAttribute) as? Bool) != true,
              windowInfos().contains(where: {
                  ($0[kCGWindowNumber as String] as? UInt32) == windowID &&
                  ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier
              }) else { throw fail("Selected window is closed, minimized, or unavailable") }
        return (window, app, rect)
    }

    func focus() throws {
        let (window, app, _) = try selected()
        app.activate(options: [])
        guard AXUIElementPerformAction(window, kAXRaiseAction as CFString) == .success else {
            throw fail("Cannot raise the selected window")
        }
        Thread.sleep(forTimeInterval: 0.15)
    }

    func snapshot() throws -> [String: Any] {
        let (window, app, rect) = try selected()
        var queue = elements(window, kAXChildrenAttribute)
        var visited: [AXUIElement] = []
        var controls: [[String: Any]] = [], texts: [String] = []
        while !queue.isEmpty && visited.count < 600 {
            let element = queue.removeFirst()
            if visited.contains(where: { CFEqual($0, element) }) { continue }
            visited.append(element)
            let role = string(element, kAXRoleAttribute)
            let subrole = string(element, kAXSubroleAttribute)
            if subrole.lowercased().contains("secure") || subrole.contains("CloseButton") ||
                subrole.contains("MinimizeButton") || subrole.contains("ZoomButton") { continue }
            if (attr(element, "AXHidden") as? Bool) == true { continue }
            queue.append(contentsOf: elements(element, kAXChildrenAttribute).prefix(200))
            guard let box = bounds(element), rect.intersects(box) else { continue }
            let name = [string(element, kAXTitleAttribute), string(element, kAXDescriptionAttribute),
                        string(element, "AXPlaceholderValue")].first(where: { !$0.isEmpty }) ?? ""
            if role == "AXStaticText" {
                let value = string(element, kAXValueAttribute)
                if texts.count < 40 { texts.append(String((value.isEmpty ? name : value).prefix(240))) }
            }
            guard let kind = allowedRoles[role], controls.count < 80 else { continue }
            let label = name.lowercased()
            if ["close", "minimize", "maximize", "关闭", "最小化", "最大化"].contains(where: { label.contains($0) }) { continue }
            var actions: CFArray?
            AXUIElementCopyActionNames(element, &actions)
            let supported = actions as? [String] ?? []
            var settable = DarwinBoolean(false)
            let canSet = kind == "Edit" &&
                AXUIElementIsAttributeSettable(element, kAXValueAttribute as CFString, &settable) == .success &&
                settable.boolValue
            let enabled = (attr(element, kAXEnabledAttribute) as? Bool) ?? false
            var guiAvailable = false
            var guiUnavailableReason = enabled ? "" : "Control is disabled"
            if enabled {
                do {
                    try foreground(CGPoint(x: box.midX, y: box.midY), target: element)
                    guiAvailable = true
                } catch { guiUnavailableReason = error.localizedDescription }
            }
            controls.append([
                "ref": refFor(element), "name": String(name.prefix(120)), "role": role,
                "control_type": kind, "enabled": enabled, "gui_available": guiAvailable,
                "gui_unavailable_reason": guiUnavailableReason,
                "rectangle": rectArray(box), "can_invoke": supported.contains(kAXPressAction),
                "can_set_text": canSet, "actions": supported.filter { $0 == kAXPressAction },
                "private_value": kind == "Edit" ? String(string(element, kAXValueAttribute).prefix(4000)) : "",
                "state_value": kind == "CheckBox" || kind == "RadioButton" ? String(describing: attr(element, kAXValueAttribute) ?? "" as CFString) : ""
            ])
        }
        // Keep only live AX objects; refs are never reassigned to another control.
        let live = Set(controls.compactMap { $0["ref"] as? String })
        refs = refs.filter { live.contains($0.0) }
        let state: [String: Any] = [
            "window_title": String(string(window, kAXTitleAttribute).prefix(160)),
            "window_token": windowToken, "window_id": windowID, "pid": app.processIdentifier,
            "bundle_id": app.bundleIdentifier ?? "", "bounds": rectArray(rect),
            "text": String(texts.joined(separator: "\n").prefix(2000)), "controls": controls
        ]
        previousState = try jsonString(state)
        return state
    }

    func foreground(_ point: CGPoint? = nil, target: AXUIElement? = nil) throws {
        let (window, app, _) = try selected()
        let ax = AXUIElementCreateApplication(app.processIdentifier)
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier,
              let focused = attr(ax, kAXFocusedWindowAttribute), CFEqual(focused, window) else {
            throw fail("Selected window is not foreground; observe and focus explicitly before retrying")
        }
        guard let point else { return }
        // CG coordinates and AX bounds are desktop points, not Retina pixels.
        var displayCount: UInt32 = 0
        guard CGGetDisplaysWithPoint(point, 0, nil, &displayCount) == .success, displayCount > 0 else {
            throw fail("GUI target is offscreen")
        }
        let infos = windowInfos()
        guard let index = infos.firstIndex(where: { ($0[kCGWindowNumber as String] as? UInt32) == windowID }) else {
            throw fail("Selected window is no longer visible")
        }
        if infos.prefix(index).contains(where: {
            (($0[kCGWindowAlpha as String] as? Double) ?? 1) > 0 &&
            infoBounds($0)?.contains(point) == true &&
            !(target != nil && isDockInteractionPlane($0, at: point))
        }) { throw fail("GUI target is obscured by another window") }
        if let target {
            var hit: AXUIElement?
            guard AXUIElementCopyElementAtPosition(AXUIElementCreateSystemWide(), Float(point.x), Float(point.y), &hit) == .success else {
                throw fail("Cannot hit-test GUI target")
            }
            for _ in 0..<12 {
                guard let current = hit else { break }
                if CFEqual(current, target) { return }
                hit = attr(current, kAXParentAttribute).map { $0 as! AXUIElement }
            }
            throw fail("GUI hit-test does not identify the observed control")
        }
    }

    func click(_ point: CGPoint) throws {
        for type in [CGEventType.mouseMoved, .leftMouseDown, .leftMouseUp] {
            guard let event = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: point, mouseButton: .left) else {
                throw fail("Cannot create mouse event")
            }
            event.post(tap: .cghidEventTap)
        }
    }

    func typeText(_ text: String, target: AXUIElement) throws {
        func checkFocus() throws {
            try foreground()
            guard let focused = attr(AXUIElementCreateApplication(app!.processIdentifier), kAXFocusedUIElementAttribute),
                  CFEqual(focused, target) else { throw fail("Text target did not acquire keyboard focus") }
        }
        try checkFocus()
        for down in [true, false] {
            let event = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: down)!
            event.flags = .maskCommand; event.post(tap: .cghidEventTap)
        }
        Thread.sleep(forTimeInterval: 0.05)
        let units = Array(text.utf16)
        var offset = 0
        while offset < units.count {
            try checkFocus()
            var end = min(offset + 20, units.count)
            if end < units.count && (0xD800...0xDBFF).contains(units[end - 1]) { end -= 1 }
            let chunk = Array(units[offset..<end])
            for down in [true, false] {
                let event = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: down)!
                // Newly constructed events can inherit the preceding Command+A
                // modifier flags. Unicode input must be unmodified text.
                event.flags = []
                event.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: chunk)
                event.post(tap: .cghidEventTap)
            }
            offset = end
        }
    }

    func act(_ request: [String: Any]) throws -> [String: Any] {
        guard let expected = previousState else { throw fail("Observe before acting") }
        // Consume freshness on every attempt, including partially delivered GUI
        // input and failed checks. The caller must observe again before retrying.
        defer { previousState = nil }
        let state = try snapshot()
        guard expected == previousState else { throw fail("AX state changed before execution") }
        guard request["window_token"] as? String == windowToken,
              let ref = request["ref"] as? String,
              let control = (state["controls"] as? [[String: Any]])?.first(where: { $0["ref"] as? String == ref }),
              let target = refs.first(where: { $0.0 == ref })?.1,
              control["enabled"] as? Bool == true,
              let operation = request["operation"] as? String,
              let route = request["route"] as? String else { throw fail("Invalid or stale AX target") }
        let value = request["value"] as? String ?? ""
        guard value.count <= 1000 else { throw fail("Text input exceeds the allowed bound") }
        if operation == "fill" {
            guard control["can_set_text"] as? Bool == true, !value.isEmpty else { throw fail("Target is not editable") }
        } else if operation != "click" || !value.isEmpty { throw fail("Unsupported AX operation") }
        if route == "ax" {
            let result: AXError
            if operation == "click" {
                guard control["can_invoke"] as? Bool == true else { throw fail("AXPress is unavailable") }
                result = AXUIElementPerformAction(target, kAXPressAction as CFString)
            } else { result = AXUIElementSetAttributeValue(target, kAXValueAttribute as CFString, value as CFString) }
            guard result == .success else { throw fail("AX action failed (\(result.rawValue))") }
        } else if route == "gui" {
            guard let box = bounds(target) else { throw fail("GUI target geometry is unavailable") }
            let point = CGPoint(x: box.midX, y: box.midY)
            try foreground(point, target: target)
            try click(point)
            if operation == "fill" {
                Thread.sleep(forTimeInterval: 0.1)
                try typeText(value, target: target)
            }
        } else { throw fail("Unsupported execution route") }
        return ["ref": ref, "operation": operation, "route": route, "window_id": windowID]
    }

    func filter() async throws -> (SCContentFilter, SCStreamConfiguration) {
        try requireCapture()
        let (_, app, _) = try selected()
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        guard let candidate = content.windows.first(where: {
            $0.windowID == windowID && $0.owningApplication?.processID == app.processIdentifier
        }) else { throw fail("Selected capture window is unavailable") }
        let filter = SCContentFilter(desktopIndependentWindow: candidate)
        let config = SCStreamConfiguration()
        config.width = Int(filter.contentRect.width * CGFloat(filter.pointPixelScale))
        config.height = Int(filter.contentRect.height * CGFloat(filter.pointPixelScale))
        guard config.width > 0, config.height > 0, config.width * config.height <= 16_000_000 else {
            throw fail("Window capture dimensions are outside bounds")
        }
        config.showsCursor = false
        config.capturesAudio = false
        config.ignoreShadowsSingleWindow = true
        config.shouldBeOpaque = true
        config.backgroundColor = captureBackground
        return (filter, config)
    }

    func capture() async throws -> [String: Any] {
        let before = try selected().2
        let (filter, config) = try await filter()
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        guard before == (try selected().2) else { throw fail("Window moved during capture") }
        let bitmap = NSBitmapImageRep(cgImage: image)
        guard let data = bitmap.representation(using: .png, properties: [:]) else { throw fail("Cannot encode window image") }
        return ["png": data.base64EncodedString(), "bounds": rectArray(before), "window_id": windowID,
                "pixel_width": image.width, "pixel_height": image.height]
    }

    func recordStart(_ request: [String: Any]) async throws -> [String: Any] {
        guard #available(macOS 15.0, *) else { throw fail("Window recording requires macOS 15 or newer") }
        try requireUnlockedSession()
        guard stream == nil, let path = request["path"] as? String, path.hasPrefix("/"), path.hasSuffix(".mp4"),
              !FileManager.default.fileExists(atPath: path) else { throw fail("Recording needs a new absolute .mp4 path") }
        let (filter, config) = try await filter()
        config.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        // H.264 requires even dimensions; the filter remains a single window.
        config.width = (config.width / 2) * 2; config.height = (config.height / 2) * 2
        let delegate = RecordingDelegate()
        let outputConfig = SCRecordingOutputConfiguration()
        outputConfig.outputURL = URL(fileURLWithPath: path)
        outputConfig.outputFileType = .mp4
        outputConfig.videoCodecType = .h264
        let output = SCRecordingOutput(configuration: outputConfig, delegate: delegate)
        let stream = SCStream(filter: filter, configuration: config, delegate: delegate)
        try stream.addRecordingOutput(output)
        self.stream = stream; recordingDelegate = delegate; recordingOutput = output
        recordingOutputAttached = true
        do {
            delegate.state.mark("start_capture_requested")
            try await boundedCapture("start_capture") { stream.startCapture(completionHandler: $0) }
            try await waitForRecording(delegate.state, finished: false, timeout: 5)
        } catch {
            // Preserve startup and cleanup failures; retained ownership also
            // lets EOF cleanup retry if the capture never confirms its stop.
            delegate.state.mark("start_failed", error: error)
            do { _ = try await recordStop() } catch { throw error }
            throw error
        }
        return ["recording": true, "window_id": windowID, "width": config.width, "height": config.height]
    }

    func recordStop() async throws -> [String: Any] {
        guard #available(macOS 15.0, *), let stream, let delegate = recordingDelegate as? RecordingDelegate,
              let output = recordingOutput as? SCRecordingOutput else {
            throw fail("No window recording is active")
        }
        let result = await finishRecording(delegate.state, remove: {
            if self.recordingOutputAttached {
                try stream.removeRecordingOutput(output)
                self.recordingOutputAttached = false
            }
        }, stop: { stream.stopCapture(completionHandler: $0) })
        if result.captureStopped {
            self.stream = nil; recordingDelegate = nil; recordingOutput = nil
            recordingOutputAttached = false
        }
        guard result.issues.isEmpty else { throw RecordingFailure(result: result) }
        return ["recording": false, "finalized": true, "events": result.events]
    }

    func editorWindows(_ request: [String: Any]) throws -> [String: Any] {
        try requireAX()
        guard let bundle = request["bundle_id"] as? String,
              ["com.apple.TextEdit", "com.microsoft.VSCode"].contains(bundle),
              let path = request["path"] as? String, path.hasPrefix("/") else {
            throw fail("Editor probe needs a registered application and absolute document path")
        }
        let expected = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        let infos = windowInfos()
        var matches: [String: String] = [:]
        for app in NSRunningApplication.runningApplications(withBundleIdentifier: bundle) {
            let ax = AXUIElementCreateApplication(app.processIdentifier)
            AXUIElementSetMessagingTimeout(ax, 1.0)
            for window in elements(ax, kAXWindowsAttribute) {
                let document = string(window, kAXDocumentAttribute)
                guard let url = URL(string: document), url.isFileURL,
                      url.standardizedFileURL.resolvingSymlinksInPath() == expected,
                      let rect = bounds(window) else { continue }
                let found = infos.filter {
                    ($0[kCGWindowOwnerPID as String] as? Int32) == app.processIdentifier &&
                    ($0[kCGWindowLayer as String] as? Int) == 0 && infoBounds($0) == rect
                }
                if found.count == 1, let id = found[0][kCGWindowNumber as String] as? UInt32 {
                    matches[String(id)] = string(window, kAXTitleAttribute)
                }
            }
        }
        return ["windows": matches]
    }

    func handle(_ request: [String: Any]) async throws -> [String: Any] {
        switch request["command"] as? String {
        case "health": return ["accessibility": AXIsProcessTrusted(),
                               "screen_recording": CGPreflightScreenCaptureAccess(),
                               "screen_locked": currentScreenLockStatus().map { $0 as Any } ?? NSNull()]
        case "select": return try select(request)
        case "observe": return try snapshot()
        case "act": return try act(request)
        case "focus": try focus(); return try snapshot()
        case "capture": return try await capture()
        case "record_start": return try await recordStart(request)
        case "record_stop": return try await recordStop()
        case "editor_windows": return try editorWindows(request)
        default: throw fail("Unknown native command")
        }
    }
}

@MainActor
final class FixtureDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let input = NSTextField(frame: NSRect(x: 30, y: 185, width: 500, height: 30))
    let status = NSTextField(labelWithString: "No note saved")
    let reviewed = NSButton(checkboxWithTitle: "Reviewed", target: nil, action: nil)
    var saved = ""
    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 200, y: 180, width: 560, height: 320),
                          styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "CUA-JEV Local Fixture"
        let heading = NSTextField(labelWithString: "CUA-JEV · macOS native test")
        heading.frame = NSRect(x: 30, y: 265, width: 500, height: 30)
        heading.font = .systemFont(ofSize: 22, weight: .semibold)
        window.contentView!.addSubview(heading)
        input.placeholderString = "Note text"; input.setAccessibilityLabel("Note text")
        input.setAccessibilityIdentifier("note-text")
        window.contentView!.addSubview(input)
        let save = NSButton(title: "Save note", target: self, action: #selector(saveNote))
        save.frame = NSRect(x: 30, y: 130, width: 130, height: 34)
        window.contentView!.addSubview(save)
        reviewed.target = self; reviewed.action = #selector(updateStatus)
        reviewed.frame = NSRect(x: 185, y: 130, width: 160, height: 34)
        window.contentView!.addSubview(reviewed)
        status.frame = NSRect(x: 30, y: 35, width: 500, height: 70)
        status.maximumNumberOfLines = 3
        window.contentView!.addSubview(status)
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }
    @objc func saveNote() { saved = input.stringValue; updateStatus() }
    @objc func updateStatus() { status.stringValue = "Saved: \(saved)\nReviewed: \(reviewed.state == .on ? "yes" : "no")" }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

#if !CUA_JEV_NATIVE_TESTS
@main
struct Main {
    @MainActor static func main() async {
        if CommandLine.arguments.contains("--fixture") {
            let application = NSApplication.shared
            application.setActivationPolicy(.regular)
            let delegate = FixtureDelegate()
            application.delegate = delegate
            application.run()
            return
        }
        let bridge = Bridge()
        while let line = await Task.detached(operation: { readLine() }).value {
            do {
                guard let data = line.data(using: .utf8), data.count <= 32_000,
                      let request = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw fail("Invalid native request")
                }
                let result = try await bridge.handle(request)
                print(try jsonString(["ok": true, "result": result]))
            } catch {
                var response: [String: Any] = ["ok": false, "error": error.localizedDescription]
                if let failure = error as? RecordingFailure { response["recording_diagnostics"] = failure.json }
                print((try? jsonString(response)) ?? "{\"ok\":false}")
            }
            fflush(stdout)
        }
        if bridge.stream != nil { _ = try? await bridge.recordStop() }
    }
}
#endif
