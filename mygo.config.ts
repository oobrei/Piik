import { defineConfig } from "mygo-cli";

// Piik App is a mygo application: the window shows the App's own loopback
// pages, which the in-process Go room service serves. The frontend is built by
// the repository's Vite pipeline and embedded by internal/server/webassets, so
// mygo embeds no frontend of its own and runs no build command; the Go binary
// is produced by scripts/assemble-app.mjs, which owns the release contract
// (capture and tunnel sidecars, license notices, REVISION).
export default defineConfig({
  name: "Piik App",
  slug: "piik-app",
  identifier: "tv.piik.app",
  copyright: "© Piik contributors",
  license: "MIT",
  main: ".",
  out: "build/app",
  macos: {
    infoPlist: {
      NSMicrophoneUsageDescription: "Use your microphone for voice communication.",
      NSCameraUsageDescription: "Share a camera selected by you.",
      NSScreenCaptureUsageDescription: "Share a screen or application selected by you.",
      NSLocalNetworkUsageDescription:
        "Connect to devices on your local network for Piik rooms and screen sharing.",
      NSAudioCaptureUsageDescription: "Share sound from the selected screen or application.",
    },
  },
});
