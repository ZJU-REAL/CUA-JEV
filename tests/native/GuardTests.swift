import CoreGraphics
import Foundation

@main
struct GuardTests {
    static func main() {
        let display = CGRect(x: 0, y: 0, width: 1728, height: 1117)
        func accepted(bundle: String? = "com.apple.dock",
                      path: String? = "/System/Library/CoreServices/Dock.app",
                      name: String? = "Dock", layer: Int? = 20,
                      bounds: CGRect? = nil) -> Bool {
            isDockInteractionPlane(bundleID: bundle, bundlePath: path, name: name,
                                   layer: layer, rect: bounds ?? display, displayBounds: [display])
        }
        precondition(accepted())
        precondition(!accepted(bundle: "example.overlay"))
        precondition(!accepted(path: "/Applications/Dock.app"))
        precondition(!accepted(name: "Launchpad"))
        precondition(!accepted(name: nil))
        precondition(!accepted(layer: 0))
        precondition(!accepted(layer: 101))
        precondition(!accepted(bounds: CGRect(x: 0, y: 1050, width: 1728, height: 67)))
        precondition(!accepted(bounds: CGRect(x: 1, y: 0, width: 1728, height: 1117)))
        precondition(!isDockInteractionPlane(bundleID: "com.apple.dock",
                      bundlePath: "/System/Library/CoreServices/Dock.app", name: "Dock",
                      layer: 20, rect: nil, displayBounds: [display]))
        print("Native occlusion guard cases passed")
    }
}
