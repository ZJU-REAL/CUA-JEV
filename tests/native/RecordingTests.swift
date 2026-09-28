import Foundation

@main
struct RecordingTests {
    @MainActor static func main() async throws {
        precondition(screenLockStatus(nil) == nil)
        precondition(screenLockStatus([:]) == nil)
        precondition(screenLockStatus(["CGSSessionScreenIsLocked": "false"]) == nil)
        precondition(screenLockStatus(["CGSSessionScreenIsLocked": 1]) == nil)
        precondition(screenLockStatus(["CGSSessionScreenIsLocked": true]) == true)
        precondition(screenLockStatus(["CGSSessionScreenIsLocked": false]) == false)
        try requireUnlockedSession(nil)
        try requireUnlockedSession(false)
        do {
            try requireUnlockedSession(true)
            preconditionFailure("locked session must fail before native window access")
        } catch { precondition(error.localizedDescription.contains("screen is locked")) }

        let state = RecordingState()
        var calls: [String] = []
        let success = await finishRecording(state, finalizeTimeout: 0.1, stopTimeout: 0.1, remove: {
            calls.append("remove")
            state.mark("finished")
        }, stop: { callback in
            precondition(state.snapshot().finished)
            calls.append("stop"); callback(nil)
        })
        precondition(calls == ["remove", "stop"] && success.captureStopped && success.issues.isEmpty)
        precondition(success.events.map { $0["event"] as! String } == [
            "remove_output_requested", "finished", "output_removed", "stop_capture_requested", "capture_stopped",
        ])

        let connectionError = NSError(domain: "RPRecordingErrorDomain", code: -5814,
                                      userInfo: [NSLocalizedDescriptionKey: "Connection invalid"])
        var cleanupCalled = false
        let removeFailed = await finishRecording(RecordingState(), remove: {
            throw connectionError
        }, stop: { callback in cleanupCalled = true; callback(nil) })
        precondition(cleanupCalled && removeFailed.captureStopped && removeFailed.issues.count == 1)
        precondition(removeFailed.issues[0].domain == "RPRecordingErrorDomain")
        precondition(removeFailed.issues[0].code == -5814)

        let before = ProcessInfo.processInfo.systemUptime
        let timedOut = await finishRecording(RecordingState(), finalizeTimeout: 0.03, stopTimeout: 0.03,
                                             remove: {}, stop: { _ in })
        precondition(ProcessInfo.processInfo.systemUptime - before < 0.5)
        precondition(!timedOut.captureStopped && timedOut.issues.map(\.stage) == ["finalize_output", "stop_capture"])
        precondition(timedOut.issues.allSatisfy { $0.domain == "CUAJEVRecordingTimeout" })

        let failed = RecordingState()
        failed.mark("recording_failed", error: connectionError)
        failed.mark("finished")
        let failureWins = await finishRecording(failed, remove: {}, stop: { callback in
            callback(NSError(domain: "StopCaptureError", code: 17))
        })
        precondition(!failureWins.captureStopped && failureWins.issues.count == 2)
        precondition(failureWins.issues.map(\.domain) == ["RPRecordingErrorDomain", "StopCaptureError"])
        let failure = RecordingFailure(result: failureWins)
        precondition(failure.localizedDescription.contains("RPRecordingErrorDomain (-5814)"))
        let diagnostic = try jsonString(failure.json)
        precondition(diagnostic.contains("recording_failed") && diagnostic.contains("\"code\":-5814"))

        // The same synchronization covers callbacks delivered concurrently by
        // different framework queues. Errors are retained even after finish.
        let concurrent = RecordingState()
        DispatchQueue.concurrentPerform(iterations: 100) { index in
            if index == 0 { concurrent.mark("started") }
            else if index == 1 { concurrent.mark("finished") }
            else { concurrent.mark("recording_failed", error: connectionError) }
            _ = concurrent.snapshot()
        }
        let snapshot = concurrent.snapshot()
        precondition(snapshot.started && snapshot.finished && snapshot.issues.count == 98)
        precondition(snapshot.events.count == 64)

        // A late framework callback after timeout is safe, and repeated
        // callbacks cannot overwrite the first completion result.
        var lateCallback: ((Error?) -> Void)?
        do {
            try await boundedCapture("late", timeout: 0.03) { lateCallback = $0 }
            preconditionFailure("missing callback must time out")
        } catch let issue as RecordingIssue { precondition(issue.stage == "late") }
        lateCallback?(nil)
        try await boundedCapture("twice", timeout: 0.03) { callback in
            callback(nil); callback(connectionError)
        }
        print("Native recording lifecycle cases passed")
    }
}
