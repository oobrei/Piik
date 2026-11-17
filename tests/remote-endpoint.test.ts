import { afterEach, describe, expect, it, vi } from "vitest";

const FINGERPRINT = "AB".repeat(32);
const COLON_FINGERPRINT = FINGERPRINT.match(/../g)!.join(":");
const GRANT = `${"A".repeat(21)}A`;
const LAUNCHER = "http://127.0.0.1:39721";
const CONNECTED = {
  connected: true,
  origin: "https://host.example",
  hostLabel: "host.example",
  certificateSha256: FINGERPRINT,
  siteAccessRequired: true,
  roomCreation: true,
  protocol: "piik-v24",
  capabilities: { sfu: false, hostRoomSession: true },
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

interface StubStorage {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
}

function storage(): StubStorage {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value);
    },
    removeItem: (key) => {
      values.delete(key);
    },
  };
}

function locationOf(href: string, base?: URL) {
  return Object.assign(new URL(href, base), { assign: vi.fn() });
}

function browser(href = `${LAUNCHER}/client`) {
  const page = {
    location: locationOf(href),
    sessionStorage: storage(),
    localStorage: storage(),
    history: {
      state: null,
      pushState: (_state: unknown, _title: string, url: string) => {
        page.location = locationOf(url, page.location);
      },
      replaceState: (_state: unknown, _title: string, url: string) => {
        page.location = locationOf(url, page.location);
      },
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  };
  vi.stubGlobal("window", page);
  return page;
}

function respond(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

/** Answers each bridge call in order, then repeats the last answer. */
function stubFetch(...answers: Array<[unknown, number?]>) {
  const calls: Array<{ path: string; init: RequestInit }> = [];
  let index = 0;
  vi.stubGlobal("fetch", vi.fn(async (path: string, init: RequestInit) => {
    calls.push({ path, init });
    const answer = answers[Math.min(index, answers.length - 1)] ?? [{}, 200];
    index += 1;
    return respond(answer[0], answer[1] ?? 200);
  }));
  return calls;
}

async function endpointModule() {
  return import("../src/client/lib/remote-endpoint");
}

describe("remote descriptor", () => {
  it("reads an invitation, a browser invite and a bare host", async () => {
    browser();
    const { parseRemoteDescriptor } = await endpointModule();

    expect(parseRemoteDescriptor(
      `piik://connect?origin=https%3A%2F%2Fhost.example&room=1234&grant=invite-grant&fp=${COLON_FINGERPRINT}`,
    )).toEqual({
      target: `piik://connect?origin=https%3A%2F%2Fhost.example&room=1234&grant=invite-grant&fp=${COLON_FINGERPRINT}`,
      origin: "https://host.example",
      roomId: "1234",
      grant: "invite-grant",
      fingerprint: FINGERPRINT,
    });

    expect(parseRemoteDescriptor(`https://host.example/r/1234#v=${GRANT}`)).toEqual({
      target: `https://host.example/r/1234#v=${GRANT}`,
      origin: "https://host.example",
      roomId: "1234",
      grant: GRANT,
    });

    expect(parseRemoteDescriptor("host.example")).toEqual({
      target: "host.example",
      origin: "https://host.example",
    });

    expect(parseRemoteDescriptor("http://127.0.0.1:8443")).toEqual({
      target: "http://127.0.0.1:8443",
      origin: "http://127.0.0.1:8443",
    });
  });

  it("rejects an address the App cannot connect to", async () => {
    browser();
    const { parseRemoteDescriptor } = await endpointModule();

    for (const value of [
      "",
      "   ",
      "piik://other?origin=host.example",
      "piik://connect?room=1234",
      "piik://connect?origin=host.example&room=12",
      "https://host.example/r/12",
      "ftp://host.example",
      "https://user:secret@host.example",
    ]) {
      expect(parseRemoteDescriptor(value), value).toBeNull();
    }
  });

  it("builds the invitation the desktop App hands over", async () => {
    browser();
    const { buildRemoteDescriptor, parseRemoteDescriptor } = await endpointModule();

    const descriptor = buildRemoteDescriptor({
      origin: "host.example",
      roomId: "1234",
      grant: GRANT,
      fingerprint: FINGERPRINT,
    });

    expect(descriptor.startsWith("piik://connect?origin=https%3A%2F%2Fhost.example")).toBe(true);
    expect(parseRemoteDescriptor(descriptor)).toEqual({
      target: descriptor,
      origin: "https://host.example",
      roomId: "1234",
      grant: GRANT,
      fingerprint: FINGERPRINT,
    });
    expect(buildRemoteDescriptor({ origin: "https://host.example" }))
      .toBe("piik://connect?origin=https%3A%2F%2Fhost.example");
  });
});

describe("remote endpoint switching", () => {
  it("keeps the page's own origin until a remote session exists", async () => {
    browser();
    const { apiPath, getRemoteEndpoint, signalWebSocketUrl, siteHostLabel } = await endpointModule();

    expect(getRemoteEndpoint()).toEqual({ session: null, viewRoute: { kind: "launcher" } });
    expect(apiPath("/api/rooms")).toBe("/api/rooms");
    expect(signalWebSocketUrl()).toBe(`ws://127.0.0.1:39721/signal`);
    expect(siteHostLabel()).toBe("127.0.0.1:39721");
  });

  it("routes API calls and signaling through the launcher proxy while remote", async () => {
    browser();
    stubFetch([CONNECTED]);
    const endpoint = await endpointModule();

    await endpoint.connectRemoteHost({ target: "host.example" });

    expect(endpoint.getRemoteEndpoint().session?.hostLabel).toBe("host.example");
    expect(endpoint.apiPath("/api/rooms/1234/access"))
      .toBe("/api/client-remote/proxy/api/rooms/1234/access");
    expect(endpoint.signalWebSocketUrl())
      .toBe("ws://127.0.0.1:39721/api/client-remote/proxy/signal");
    expect(endpoint.siteHostLabel()).toBe("host.example");
  });

  it("restores the surface the launcher URL names", async () => {
    browser(`${LAUNCHER}/client?remote=viewer&room=1234&debug=1`);
    const { getRemoteEndpoint } = await endpointModule();

    expect(getRemoteEndpoint().viewRoute).toEqual({ kind: "viewer", roomId: "1234" });
  });

  it("returns to the launcher when the session is gone", async () => {
    const page = browser(`${LAUNCHER}/client?remote=host`);
    stubFetch([{ connected: false }]);
    const endpoint = await endpointModule();

    expect(endpoint.getRemoteEndpoint().viewRoute).toEqual({ kind: "host" });
    const status = await endpoint.fetchRemoteStatus();

    expect(status.connected).toBe(false);
    expect(endpoint.getRemoteEndpoint()).toEqual({ session: null, viewRoute: { kind: "launcher" } });
    expect(page.location.pathname).toBe("/client");
    expect(page.location.search).toBe("");
  });
});

describe("in-app remote navigation", () => {
  it("keeps an invitation grant for this tab only when it is valid", async () => {
    const page = browser();
    const { storeViewerGrant } = await import("../src/client/lib/session");

    expect(storeViewerGrant("1234", GRANT)).toBe(true);
    expect(page.sessionStorage.getItem("piik:viewer-grant:1234")).toBe(GRANT);
    expect(storeViewerGrant("1234", "not-a-grant")).toBe(false);
    expect(storeViewerGrant("12", GRANT)).toBe(false);
    expect(page.sessionStorage.getItem("piik:viewer-grant:1234")).toBe(GRANT);
  });

  it("navigates the document while the App is local", async () => {
    const page = browser();
    const assign = page.location.assign;
    const { navigateAppRoute } = await import("../src/client/lib/session");

    navigateAppRoute("/r/1234");

    expect(assign).toHaveBeenCalledWith("/r/1234");
    expect(page.location.pathname).toBe("/client");
  });

  it("enters a remote room in place without leaving the launcher", async () => {
    const page = browser(`${LAUNCHER}/client?debug=1`);
    const assign = page.location.assign;
    stubFetch([CONNECTED]);
    const endpoint = await endpointModule();
    const session = await import("../src/client/lib/session");

    await endpoint.connectRemoteHost({ target: "host.example" });
    session.navigateAppRoute(`/r/1234#v=${GRANT}`);

    expect(assign).not.toHaveBeenCalled();
    expect(page.location.pathname).toBe("/client");
    expect(page.location.searchParams.get("remote")).toBe("viewer");
    expect(page.location.searchParams.get("room")).toBe("1234");
    expect(page.location.searchParams.get("debug")).toBe("1");
    expect(page.location.hash).toBe("");
    expect(page.sessionStorage.getItem("piik:viewer-grant:1234")).toBe(GRANT);
    expect(session.readViewerRouteForRoom("1234")).toEqual({ roomId: "1234", viewerGrant: GRANT });
    expect(endpoint.getRemoteEndpoint().viewRoute).toEqual({ kind: "viewer", roomId: "1234" });

    session.navigateAppRoute("/join");
    expect(page.location.searchParams.get("remote")).toBe("join");
    expect(page.location.searchParams.get("room")).toBeNull();

    endpoint.returnToRemoteLauncher();
    expect(page.location.pathname).toBe("/client");
    expect(page.location.searchParams.get("remote")).toBeNull();
    expect(endpoint.getRemoteEndpoint().viewRoute).toEqual({ kind: "launcher" });
  });
});

describe("remote bridge calls", () => {
  it("probes, connects and disconnects through the launcher", async () => {
    const page = browser();
    const calls = stubFetch(
      [{ origin: "https://host.example", hostLabel: "host.example", tlsStatus: "untrusted", certificateSha256: FINGERPRINT, info: { siteAccessRequired: true } }],
      [CONNECTED],
      [{ connected: false }],
    );
    const endpoint = await endpointModule();

    const probe = await endpoint.probeRemoteHost("host.example");
    expect(probe.tlsStatus).toBe("untrusted");
    expect(calls[0]!.path).toBe("/api/client-remote/probe");
    expect(calls[0]!.init.method).toBe("POST");
    expect(calls[0]!.init.headers).toEqual({
      Accept: "application/json",
      "Content-Type": "application/json",
    });
    expect(JSON.parse(String(calls[0]!.init.body))).toEqual({ target: "host.example" });

    const status = await endpoint.connectRemoteHost({
      target: "host.example",
      acceptedFingerprint: FINGERPRINT,
      sitePassword: "site-passphrase",
    });
    expect(status.connected).toBe(true);
    expect(calls[1]!.path).toBe("/api/client-remote/connect");
    expect(JSON.parse(String(calls[1]!.init.body))).toEqual({
      target: "host.example",
      acceptedFingerprint: FINGERPRINT,
      sitePassword: "site-passphrase",
    });

    await endpoint.disconnectRemoteHost();
    expect(calls[2]!.path).toBe("/api/client-remote/disconnect");
    expect(JSON.parse(String(calls[2]!.init.body))).toEqual({});
    expect(endpoint.getRemoteEndpoint()).toEqual({ session: null, viewRoute: { kind: "launcher" } });
    expect(page.location.search).toBe("");
  });

  it("reports a rejected site password as an unauthorized ApiError", async () => {
    browser();
    stubFetch([{ error: "the remote host rejected the site password" }, 401]);
    const endpoint = await endpointModule();
    const { ApiError } = await import("../src/client/lib/api");

    await expect(endpoint.connectRemoteHost({ target: "host.example", sitePassword: "wrong" }))
      .rejects.toBeInstanceOf(ApiError);
    await expect(endpoint.connectRemoteHost({ target: "host.example", sitePassword: "wrong" }))
      .rejects.toMatchObject({ status: 401 });
    expect(endpoint.getRemoteEndpoint().session).toBeNull();
  });

  it("records the host the user allowed", async () => {
    browser();
    const calls = stubFetch([{ remoteHosts: [] }]);
    const endpoint = await endpointModule();

    await endpoint.setRemoteHostSaved("https://host.example", FINGERPRINT, true);

    expect(calls[0]!.path).toBe("/api/client-remote/host");
    expect(JSON.parse(String(calls[0]!.init.body))).toEqual({
      origin: "https://host.example",
      fingerprint: FINGERPRINT,
      allowed: true,
    });
  });
});
