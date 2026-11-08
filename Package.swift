// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXComfyUI",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MLXDiffusionCore", targets: ["MLXDiffusionCore"]),
        .library(name: "QwenImage21", targets: ["QwenImage21"]),
        .library(name: "LTX2", targets: ["LTX2"]),
        .library(name: "GraphRuntime", targets: ["GraphRuntime"]),
        .library(name: "NodesStandard", targets: ["NodesStandard"]),
        .executable(name: "ltx-probe-text", targets: ["ltx-probe-text"]),
        .executable(name: "ltx-probe-dit", targets: ["ltx-probe-dit"]),
        .executable(name: "ltx-probe-vae", targets: ["ltx-probe-vae"]),
        .executable(name: "ltx-probe-audio", targets: ["ltx-probe-audio"]),
        .executable(name: "mlxcomfy-cli", targets: ["mlxcomfy-cli"]),
        .executable(name: "MLXComfyApp", targets: ["MLXComfyApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.33.0")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.0"),
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
        .target(name: "GraphRuntime"),
        .target(
            name: "NodesStandard",
            dependencies: [
                "GraphRuntime",
                "MLXDiffusionCore",
                "QwenImage21",
                "LTX2",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "mlxcomfy-cli",
            dependencies: [
                "MLXDiffusionCore",
                "LTX2",
                "QwenImage21",
                "GraphRuntime",
                "NodesStandard",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXComfyApp",
            dependencies: [
                "MLXDiffusionCore",
                "GraphRuntime",
                "NodesStandard",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // AVKit's SwiftUI VideoPlayer needs the framework linked explicitly in a SwiftPM executable.
            linkerSettings: [.linkedFramework("AVKit"), .linkedFramework("AVFoundation")]
        ),
        .testTarget(
            name: "MLXDiffusionCoreTests",
            dependencies: ["MLXDiffusionCore", "QwenImage21"]
        ),
        .testTarget(
            name: "LTX2Tests",
            dependencies: ["LTX2", "MLXDiffusionCore", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "GraphRuntimeTests",
            dependencies: ["GraphRuntime"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
