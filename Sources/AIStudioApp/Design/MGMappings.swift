import GraphRuntime
import AIStudioLibrary
import NodesRemote
import SwiftUI

// The app's colour-coded things on the design system's palette hues: one hue per data type (sockets and wires),
// per node category, per group colour and per preset category.

extension SocketType {
    var color: Color {
        switch self {
        case .model: return MG.huePurple
        case .conditioning: return MG.hueOrange
        case .latent: return MG.huePink
        case .image: return MG.hueBlue
        case .mask: return MG.hueGreen
        case .string: return MG.hueMint
        case .video: return MG.hueRed
        case .audio: return MG.hueTeal
        case .cloudLoRAs: return MG.huePurple  // LoRAs, like the model they change
        default: return MG.hueGray
        }
    }
}

extension NodeSchema {
    var tint: Color {
        switch category {
        case "loaders": return MG.huePurple
        case "conditioning": return MG.hueOrange
        case "latent": return MG.huePink
        case "sampling": return MG.hueIndigo
        case "image": return MG.hueBlue
        case "subgraph": return MG.huePurple
        case "utils": return MG.hueBrown
        case "video": return MG.hueRed
        case "audio": return MG.hueTeal
        default: return MG.hueGray
        }
    }
}

extension GraphGroup {
    var tint: Color {
        switch color {
        case "green": return MG.hueGreen
        case "yellow": return MG.hueYellow
        case "red": return MG.hueRed
        case "purple": return MG.huePurple
        case "gray": return MG.hueGray
        default: return MG.hueBlue
        }
    }
}

extension Preset.Category {
    var hue: Color {
        switch self {
        case .image: return MG.hueBlue
        case .video: return MG.hueRed
        case .sound: return MG.hueTeal
        case .music: return MG.huePink
        case .speech: return MG.hueIndigo
        }
    }
}
