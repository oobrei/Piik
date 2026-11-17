import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import {
  RemoteConnectView,
  type RemoteConnectViewProps,
} from "../src/client/components/living/RemoteConnectCard";
import { RemoteSessionMenu } from "../src/client/components/living/Header";
import { setCopy } from "../src/client/ui/copy";
import type {
  RemoteHost,
  RemoteProbeResult,
  RemoteSessionStatus,
} from "../src/client/lib/remote-endpoint";

const FINGERPRINT = "A".repeat(64);
const site = "https://site.example";

const trusted: RemoteProbeResult = {
  origin: site,
  hostLabel: "site.example",
  tlsStatus: "trusted",
  certificateSha256: FINGERPRINT,
  info: { siteAccessRequired: false, roomCreation: true },
};
const untrusted: RemoteProbeResult = { ...trusted, tlsStatus: "untrusted" };
const session: RemoteSessionStatus = {
  connected: true,
  origin: site,
  hostLabel: "site.example",
  certificateSha256: FINGERPRINT,
  siteAccessRequired: false,
  roomCreation: true,
};

function view(overrides: Partial<RemoteConnectViewProps> = {}): string {
  return renderToStaticMarkup(createElement(RemoteConnectView, {
    session: null,
    remoteHosts: [],
    state: { kind: "idle" },
    address: "",
    passphrase: "",
    passphraseRef: { current: null },
    rememberHost: false,
    needsPassphrase: false,
    error: null,
    busy: false,
    onAddressChange() {},
    onPassphraseChange() {},
    onRememberHostChange() {},
    onSubmit() {},
    onTrust() {},
    onCancel() {},
    onOpen() {},
    onConnect() {},
    onConnectSaved() {},
    onRemoveHost() {},
    onDisconnect() {},
    ...overrides,
  } as RemoteConnectViewProps));
}

// Language and presentation are session state; fix both for readable assertions.
afterEach(() => setCopy({ lang: "en", vis: false }));

describe("remote App connection", () => {
  it("offers one address form and the sites the user already allowed", () => {
    setCopy({ lang: "en", vis: false });
    const hosts: RemoteHost[] = [
      { origin: "https://pinned.example", fingerprint: FINGERPRINT },
      { origin: "https://open.example" },
    ];
    const html = view({ remoteHosts: hosts });
    expect(html.match(/<form/g)).toHaveLength(1);
    expect(html).toContain('aria-label="Remote connection"');
    expect(html).toContain("https://pinned.example");
    expect(html).toContain("https://open.example");
    expect(html).toContain('aria-label="Remove this saved site"');
    // A pinned certificate is marked differently from a site without one.
    const saved = html.slice(html.indexOf("lr-client-remote-saved\""), html.indexOf("lr-client-remote-form"));
    const icons = [...saved.matchAll(/<svg[\s\S]*?<\/svg>/g)].map(match => match[0]);
    expect(icons).toHaveLength(4);
    expect(icons[0]).not.toBe(icons[2]);
  });

  it("keeps the visual mode on icons and hidden labels", () => {
    setCopy({ lang: "en", vis: true });
    const html = view({ remoteHosts: [{ origin: site, fingerprint: FINGERPRINT }] });
    expect(html).toContain('aria-label="Connect"');
    expect(html).toContain('aria-label="Site address or invitation"');
    expect(html).toContain("visually-hidden");
    expect(html).not.toContain('class="lr-cap"');
  });

  it("requires an explicit trust click before an untrusted certificate is used", () => {
    setCopy({ lang: "en", vis: false });
    const html = view({ state: { kind: "untrusted", probe: untrusted, descriptor: { target: site, origin: site } } });
    expect(html).toContain("This site&#x27;s certificate is not trusted yet");
    expect(html).toContain("SHA-256 fingerprint");
    expect(html).toContain(FINGERPRINT);
    expect(html).toMatch(/<code dir="ltr">/);
    expect(html).toContain('aria-label="Trust this certificate"');
    expect(html).toContain('aria-label="Cancel"');
    // No surface may be entered before that click.
    expect(html).not.toContain('aria-label="Join a room"');
    expect(html).not.toContain('aria-label="Create a room"');
  });

  it("joins the invited room instead of offering another surface", () => {
    setCopy({ lang: "en", vis: false });
    const html = view({
      state: { kind: "trusted", probe: trusted, descriptor: { target: site, origin: site, roomId: "1234", grant: "invite" } },
    });
    expect(html).toContain("site.example");
    expect(html).toContain('aria-label="Join room 1234"');
    expect(html).toContain('aria-label="Remember this site"');
    expect(html).not.toContain('aria-label="Create a room"');
    expect(html).not.toContain('type="password"');
  });

  it("offers the surfaces the trusted site supports", () => {
    setCopy({ lang: "en", vis: false });
    const html = view({ state: { kind: "trusted", probe: trusted, descriptor: { target: site, origin: site } } });
    expect(html).toContain('aria-label="Join a room"');
    expect(html).toContain('aria-label="Create a room"');
    const restricted = view({
      state: {
        kind: "trusted",
        probe: { ...trusted, info: { siteAccessRequired: true, roomCreation: false } },
        descriptor: { target: site, origin: site },
      },
    });
    expect(restricted).toContain('type="password"');
    expect(restricted).toContain('aria-label="Site passphrase"');
    expect(restricted).not.toContain('aria-label="Create a room"');
  });

  it("asks for the passphrase a rejected connection revealed", () => {
    setCopy({ lang: "en", vis: false });
    const html = view({
      state: { kind: "trusted", probe: trusted, descriptor: { target: site, origin: site } },
      needsPassphrase: true,
      error: "The site passphrase is incorrect",
    });
    expect(html).toContain('role="alert"');
    expect(html).toContain("The site passphrase is incorrect");
    expect(html).toContain('type="password"');
    expect(html).toContain('aria-label="Site passphrase"');
  });

  it("shows the connected session with its own launcher actions", () => {
    setCopy({ lang: "en", vis: false });
    const html = view({ session });
    expect(html).toContain('aria-label="Remote session"');
    expect(html).toContain("site.example");
    expect(html).toContain('aria-label="Create a room"');
    expect(html).toContain('aria-label="Join a room"');
    expect(html).toContain('aria-label="Disconnect"');
    expect(html.match(/<form/g)).toBeNull();
  });

  it("keeps the header session control on the launcher origin", () => {
    setCopy({ lang: "en", vis: false });
    const html = renderToStaticMarkup(createElement(RemoteSessionMenu, { session }));
    expect(html).toContain("lr-remote-session-card");
    expect(html).toContain('popover="auto"');
    expect(html).toContain('aria-haspopup="dialog"');
    expect(html).toContain('aria-label="Remote session"');
    expect(html).toContain("site.example");
    expect(html).toContain("Back to App launcher");
    expect(html).toContain('aria-label="Disconnect"');
    // The popover names the certificate without printing the whole value.
    expect(html).toContain(`${FINGERPRINT.slice(0, 16)}…`);
    expect(html).not.toContain(FINGERPRINT);
  });

  it("renders the remote card beside the launcher form, never inside it", () => {
    const source = readFileSync(
      new URL("../src/client/pages/AppLauncherPage.tsx", import.meta.url), "utf8",
    );
    const stack = source.indexOf('<div className="lr-client-launch-stack">');
    const formClose = source.indexOf("</LauncherForm>");
    const card = source.indexOf("<RemoteConnectCard");
    expect(stack).toBeGreaterThan(-1);
    expect(formClose).toBeGreaterThan(stack);
    expect(card).toBeGreaterThan(formClose);
  });
});
