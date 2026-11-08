// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXComfyUI",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MLXDiffusionCore", targets: ["MLXDiffusionCore"]),
        .library(name: "QwenImage21", targets: ["QwenImage21"]),
        .library(name: "GraphRuntime", targets: ["GraphRuntime"]),
        .library(name: "NodesStandard", targets: ["NodesStandard"]),
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
        .target(name: "GraphRuntime"),
        .target(
            name: "NodesStandard",
            dependencies: [
                "GraphRuntime",
                "MLXDiffusionCore",
                "QwenImage21",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "mlxcomfy-cli",
            dependencies: [
                "MLXDiffusionCore",
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
            ]
        ),
        .testTarget(
            name: "MLXDiffusionCoreTests",
            dependencies: ["MLXDiffusionCore", "QwenImage21"]
        ),
        .testTarget(
            name: "GraphRuntimeTests",
            dependencies: ["GraphRuntime"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
