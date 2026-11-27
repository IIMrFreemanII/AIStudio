// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AIStudio",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MLXDiffusionCore", targets: ["MLXDiffusionCore"]),
        .library(name: "QwenImage21", targets: ["QwenImage21"]),
        .library(name: "LTX2", targets: ["LTX2"]),
        .library(name: "StableAudio", targets: ["StableAudio"]),
        .library(name: "ACEStep", targets: ["ACEStep"]),
        .library(name: "KokoroSpeech", targets: ["KokoroSpeech"]),
        .library(name: "Wan", targets: ["Wan"]),
        .library(name: "Flux", targets: ["Flux"]),
        .library(name: "GraphRuntime", targets: ["GraphRuntime"]),
        .library(name: "NodesStandard", targets: ["NodesStandard"]),
        .library(name: "AIStudioProtocol", targets: ["AIStudioProtocol"]),
        .library(name: "NodesRemote", targets: ["NodesRemote"]),
        .library(name: "LLMClient", targets: ["LLMClient"]),
        .library(name: "Agent", targets: ["Agent"]),
        .library(name: "MovieCore", targets: ["MovieCore"]),
        .library(name: "CharacterCore", targets: ["CharacterCore"]),
        .library(name: "AIStudioServer", targets: ["AIStudioServer"]),
        .executable(name: "ltx-probe-text", targets: ["ltx-probe-text"]),
        .executable(name: "ltx-probe-dit", targets: ["ltx-probe-dit"]),
        .executable(name: "ltx-probe-vae", targets: ["ltx-probe-vae"]),
        .executable(name: "ltx-probe-audio", targets: ["ltx-probe-audio"]),
        .executable(name: "ace-probe-text", targets: ["ace-probe-text"]),
        .executable(name: "ace-probe-dit", targets: ["ace-probe-dit"]),
        .executable(name: "ace-probe-lm", targets: ["ace-probe-lm"]),
        .executable(name: "ace-probe-vae", targets: ["ace-probe-vae"]),
        .executable(name: "kokoro-probe-g2p", targets: ["kokoro-probe-g2p"]),
        .executable(name: "kokoro-probe-model", targets: ["kokoro-probe-model"]),
        .executable(name: "wan-probe", targets: ["wan-probe"]),
        .executable(name: "flux-probe", targets: ["flux-probe"]),
        .executable(name: "aistudio-cli", targets: ["aistudio-cli"]),
        .executable(name: "AIStudioApp", targets: ["AIStudioApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.33.0")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.0"),
        // The remote server (`aistudio-cli serve`) only; the app talks to it with URLSession.
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.28.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird-websocket.git", from: "2.8.0"),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.80.0"),
        .package(url: "https://github.com/apple/swift-http-types.git", from: "1.3.0"),
        // The coding chat's terminal (a terminal emulator on a pseudo-terminal). 1.19 is the last release before the
        // 2.0 I/O rework (the 1.99 previews).
        .package(url: "https://github.com/migueldeicaza/SwiftTerm.git", .upToNextMinor(from: "1.19.0")),
        // Syntax highlighting (CodeEditing): tree-sitter and its grammars, pinned exactly. swift-tree-sitter has a stray
        // 0.25.0 tag that sorts above 0.10.0, so never use `from:` for it.
        .package(url: "https://github.com/tree-sitter/swift-tree-sitter", exact: "0.10.0"),
        .package(url: "https://github.com/tree-sitter/tree-sitter", exact: "0.25.10"),
        .package(url: "https://github.com/alex-pinkus/tree-sitter-swift", exact: "0.7.4-with-generated-files"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-typescript", exact: "0.23.2"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-json", exact: "0.24.8"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-bash", exact: "0.25.1"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-c", exact: "0.24.2"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-cpp", exact: "0.23.4"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-rust", exact: "0.24.2"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-go", exact: "0.25.0"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-html", exact: "0.23.2"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-toml", exact: "0.7.0"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-markdown", exact: "0.5.4"),
        // python, javascript, css and yaml, vendored: see Packages/TreeSitterGrammars/Package.swift for why.
        .package(path: "Packages/TreeSitterGrammars"),
    ],
    targets: [
        .target(
            name: "MLXDiffusionCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "QwenImage21",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "LTX2Text",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Development probe: runs LTX2Text on dumped reference inputs and reports the error.
        .executableTarget(
            name: "ltx-probe-text",
            dependencies: ["LTX2Text", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "LTX2Transformer",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        // Development probe: runs LTX2Transformer on dumped reference inputs and reports the error.
        .executableTarget(
            name: "ltx-probe-dit",
            dependencies: ["LTX2Transformer", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "LTX2VAE",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        // Development probe: runs LTX2VAE on dumped reference inputs and reports the error.
        .executableTarget(
            name: "ltx-probe-vae",
            dependencies: ["LTX2VAE", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "LTX2Audio",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        // Development probe: runs LTX2Audio on dumped reference inputs and reports the error.
        .executableTarget(
            name: "ltx-probe-audio",
            dependencies: ["LTX2Audio", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        // LTX-2.3: text/image-to-video with a synchronized soundtrack (distilled two-stage pipeline).
        .target(
            name: "LTX2",
            dependencies: [
                "MLXDiffusionCore", "LTX2Text", "LTX2Transformer", "LTX2VAE", "LTX2Audio",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Stable Audio Open Small: text-to-sound (T5 + DiT + Oobleck VAE).
        .target(
            name: "StableAudio",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // ACE-Step 1.5 (music): Qwen3 encoders and the condition encoder.
        .target(
            name: "ACEStepText",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Development probe: runs ACEStepText on dumped reference inputs and reports the error.
        .executableTarget(
            name: "ace-probe-text",
            dependencies: ["ACEStepText", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "Tokenizers", package: "swift-transformers")]
        ),
        // ACE-Step 1.5: the diffusion transformer and its sampler.
        .target(
            name: "ACEStepDiT",
            dependencies: [
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "ace-probe-dit",
            dependencies: ["ACEStepDiT", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        // ACE-Step 1.5: the 5 Hz planner LM (song blueprint and audio codes).
        .target(
            name: "ACEStepLM",
            dependencies: [
                "ACEStepText",
                "MLXDiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "ace-probe-lm",
            dependencies: ["ACEStepLM", "ACEStepText", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .executableTarget(
            name: "ace-probe-vae",
            dependencies: ["StableAudio", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        // ACE-Step 1.5: text/lyrics-to-music pipeline (planner → DiT → Oobleck VAE at 48 kHz).
        .target(
            name: "ACEStep",
            dependencies: [
                "MLXDiffusionCore", "ACEStepText", "ACEStepDiT", "ACEStepLM", "StableAudio",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Kokoro-82M (speech): pure-Swift English G2P (misaki lexicons, no espeak) …
        .target(
            name: "KokoroG2P",
            resources: [.copy("Lexicons")]
        ),
        .executableTarget(name: "kokoro-probe-g2p", dependencies: ["KokoroG2P"]),
        // … and the StyleTTS2/iSTFTNet model with the speech pipeline.
        .target(
            name: "Kokoro",
            dependencies: [
                "MLXDiffusionCore", "KokoroG2P",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "KokoroSpeech",
            dependencies: ["Kokoro", "KokoroG2P", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .executableTarget(
            name: "kokoro-probe-model",
            dependencies: ["Kokoro", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        // T5 v1.1 / umT5 XXL text encoder (Wan's umT5, Flux's T5).
        .target(
            name: "T5Text",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        // Wan 2.2 A14B (video): the 14B DiT experts, the Wan 2.1 VAE and the pipeline.
        .target(
            name: "WanTransformer",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "WanVAE",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "Wan",
            dependencies: [
                "MLXDiffusionCore", "T5Text", "WanTransformer", "WanVAE",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        // Development probe: runs the Wan modules on dumped reference inputs and reports the error.
        .executableTarget(
            name: "wan-probe",
            dependencies: ["Wan", "T5Text", "WanTransformer", "WanVAE", "MLXDiffusionCore",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
        // FLUX.1 [dev] (images): the CLIP-L text encoder, the MMDiT transformer, the 16-channel VAE and the pipeline.
        .target(
            name: "CLIPText",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "FluxTransformer",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "FluxVAE",
            dependencies: ["MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .target(
            name: "Flux",
            dependencies: [
                "MLXDiffusionCore", "T5Text", "CLIPText", "FluxTransformer", "FluxVAE",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Development probe: runs the Flux modules on dumped reference inputs and reports the error.
        .executableTarget(
            name: "flux-probe",
            dependencies: ["Flux", "CLIPText", "FluxTransformer", "FluxVAE", "T5Text", "MLXDiffusionCore",
                           .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "Tokenizers", package: "swift-transformers")]
        ),
        .target(name: "GraphRuntime"),
        .target(
            name: "AIStudioLibrary",
            dependencies: ["GraphRuntime"],
            resources: [.copy("Presets")]
        ),
        .target(
            name: "NodesStandard",
            dependencies: [
                "GraphRuntime",
                "MLXDiffusionCore",
                "QwenImage21",
                "LTX2",
                "StableAudio",
                "ACEStep",
                "KokoroSpeech",
                "Wan",
                "Flux",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Nodes that run models on a GPU server (server/gpu, on Modal or a GPU box) and return local values.
        .target(
            name: "NodesRemote",
            dependencies: [
                "GraphRuntime", "MLXDiffusionCore", "NodesStandard", "AIStudioProtocol",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Wire format shared by the remote server, the app and the tests (no MLX, no networking stack).
        .target(
            name: "AIStudioProtocol",
            dependencies: ["GraphRuntime"]
        ),
        // Chats with LLMs on a GPU server: OpenAI-compatible streaming (SSE), the server's /v1/llm routes, transcripts.
        .target(
            name: "LLMClient",
            dependencies: ["AIStudioProtocol"]
        ),
        // The coding agent: tools that read, search, edit and run things in a project on this Mac, permission modes,
        // and the loop that drives a model's tool calls.
        .target(
            name: "Agent",
            dependencies: ["LLMClient"]
        ),
        // Movies: script, cast and shots, the LLM planner, render templates and cost estimates (no UI).
        .target(
            name: "MovieCore",
            dependencies: ["GraphRuntime", "LLMClient", "Agent"]
        ),
        // Character chat: companion characters, memory, prompts, media tags and storage (no UI).
        .target(
            name: "CharacterCore",
            dependencies: ["MovieCore", "LLMClient", "Agent", "GraphRuntime"],
            // The prompt engine's rule book: a copy of docs/prompting.md (PromptEngineTests checks they match).
            resources: [.copy("Resources/prompting.md")]
        ),
        // Drives other Mac apps for the coding agent's UI testing: accessibility, CGEvent, ScreenCaptureKit, AVFoundation.
        .target(
            name: "UIControl",
            dependencies: ["Agent"],
            linkerSettings: [.linkedFramework("ScreenCaptureKit"), .linkedFramework("ApplicationServices"),
                             .linkedFramework("AVFoundation"), .linkedFramework("CoreMedia"),
                             .linkedFramework("AppKit"), .linkedFramework("CoreGraphics")]
        ),
        // `aistudio-cli serve`: runs graphs for remote clients, plus the web dashboard.
        .target(
            name: "AIStudioServer",
            dependencies: [
                "GraphRuntime",
                "NodesStandard",
                "MLXDiffusionCore",
                "AIStudioProtocol",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
            ],
            resources: [.copy("Resources/dashboard")],
            linkerSettings: [.linkedFramework("IOKit")]
        ),
        .executableTarget(
            name: "aistudio-cli",
            dependencies: [
                "MLXDiffusionCore",
                "LTX2",
                "StableAudio",
                "ACEStep",
                "KokoroSpeech",
                "Wan",
                "Flux",
                "FluxVAE",
                "QwenImage21",
                "GraphRuntime",
                "NodesStandard",
                "NodesRemote",
                "AIStudioProtocol",
                "AIStudioServer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Tree-sitter syntax highlighting: a grammar registry and a highlighter that returns coloured spans.
        .target(
            name: "CodeEditing",
            dependencies: [
                .product(name: "SwiftTreeSitter", package: "swift-tree-sitter"),
                .product(name: "SwiftTreeSitterLayer", package: "swift-tree-sitter"),
                .product(name: "TreeSitter", package: "tree-sitter"),
                .product(name: "TreeSitterSwift", package: "tree-sitter-swift"),
                .product(name: "TreeSitterPython", package: "TreeSitterGrammars"),
                .product(name: "TreeSitterJavaScript", package: "TreeSitterGrammars"),
                .product(name: "TreeSitterTypeScript", package: "tree-sitter-typescript"),
                .product(name: "TreeSitterJSON", package: "tree-sitter-json"),
                .product(name: "TreeSitterBash", package: "tree-sitter-bash"),
                .product(name: "TreeSitterC", package: "tree-sitter-c"),
                .product(name: "TreeSitterCPP", package: "tree-sitter-cpp"),
                .product(name: "TreeSitterRust", package: "tree-sitter-rust"),
                .product(name: "TreeSitterGo", package: "tree-sitter-go"),
                .product(name: "TreeSitterHTML", package: "tree-sitter-html"),
                .product(name: "TreeSitterCSS", package: "TreeSitterGrammars"),
                .product(name: "TreeSitterYAML", package: "TreeSitterGrammars"),
                .product(name: "TreeSitterTOML", package: "tree-sitter-toml"),
                .product(name: "TreeSitterMarkdown", package: "tree-sitter-markdown"),
            ]
        ),
        .executableTarget(
            name: "AIStudioApp",
            dependencies: [
                "MLXDiffusionCore",
                "GraphRuntime",
                "NodesStandard",
                "NodesRemote",
                "AIStudioLibrary",
                "AIStudioProtocol",
                "ModelImport",
                "LLMClient",
                "Agent",
                "MovieCore",
                "CharacterCore",
                "UIControl",
                "CodeEditing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "SwiftTerm", package: "SwiftTerm"),
            ],
            // JetBrains Mono (OFL), the design system's typeface; the creator's option example photos (made with
            // OptionImageMaker).
            resources: [.copy("Resources/Fonts"), .copy("Resources/OptionImages")],
            // AVKit's SwiftUI VideoPlayer (and the chat's PDF and Quick Look previews) need their frameworks linked
            // explicitly in a SwiftPM executable.
            linkerSettings: [.linkedFramework("AVKit"), .linkedFramework("AVFoundation"), .linkedFramework("WebKit"),
                             .linkedFramework("PDFKit"), .linkedFramework("QuickLookUI")]
        ),
        .testTarget(
            name: "CodeEditingTests",
            dependencies: ["CodeEditing"]
        ),
        .testTarget(
            name: "MLXDiffusionCoreTests",
            dependencies: ["MLXDiffusionCore", "QwenImage21"]
        ),
        .testTarget(
            name: "StableAudioTests",
            dependencies: ["StableAudio", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "KokoroG2PTests",
            dependencies: ["KokoroG2P"]
        ),
        .testTarget(
            name: "KokoroSpeechTests",
            dependencies: ["KokoroSpeech", "KokoroG2P"]
        ),
        .testTarget(
            name: "ACEStepTests",
            dependencies: ["ACEStep", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "WanTests",
            dependencies: ["Wan", "WanTransformer", "T5Text", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "FluxTests",
            dependencies: ["Flux", "FluxTransformer", "CLIPText", "MLXDiffusionCore",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "LTX2Tests",
            dependencies: ["LTX2", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "GraphRuntimeTests",
            dependencies: ["GraphRuntime"]
        ),
        .testTarget(
            name: "NodesRemoteTests",
            dependencies: ["NodesRemote", "NodesStandard", "GraphRuntime", "MLXDiffusionCore", "AIStudioProtocol",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "AgentTests",
            dependencies: ["Agent", "LLMClient"]
        ),
        .testTarget(
            name: "MovieCoreTests",
            dependencies: ["MovieCore", "GraphRuntime", "LLMClient", "Agent"]
        ),
        .testTarget(
            name: "CharacterCoreTests",
            dependencies: ["CharacterCore", "LLMClient", "Agent"]
        ),
        .testTarget(
            name: "UIControlTests",
            dependencies: ["UIControl", "Agent"]
        ),
        .testTarget(
            name: "LLMClientTests",
            dependencies: ["LLMClient", "AIStudioProtocol"]
        ),
        .testTarget(
            name: "AIStudioProtocolTests",
            dependencies: ["AIStudioProtocol", "GraphRuntime", "NodesStandard"]
        ),
        .testTarget(
            name: "AIStudioServerTests",
            dependencies: [
                "AIStudioServer", "AIStudioProtocol", "GraphRuntime", "NodesStandard",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdTesting", package: "hummingbird"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
            ]
        ),
        .testTarget(
            name: "AIStudioLibraryTests",
            dependencies: ["AIStudioLibrary", "GraphRuntime", "NodesStandard", "NodesRemote"]
        ),
        // Converts fine-tuned transformers (Civitai single files, diffusers folders) into the local model layouts.
        .target(
            name: "ModelImport",
            dependencies: ["MLXDiffusionCore", "FluxTransformer", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "ModelImportTests",
            dependencies: ["ModelImport", "Flux", "FluxTransformer", "QwenImage21", "WanTransformer", "MLXDiffusionCore",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
    ],
    swiftLanguageModes: [.v5]
)
