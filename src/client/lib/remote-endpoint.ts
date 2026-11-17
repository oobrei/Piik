import { call, callWith, isCallError, isMyGo } from "mygo-runtime";
import { useSyncExternalStore } from "react";
import { z } from "zod";
import { runtimeCapabilitiesSchema } from "../../shared/protocol";
import { say } from "../ui/copy";
import { ApiError } from "./api-error";
import { activateClient, isValidRoomId } from "./session";

// The launcher owns both the proxied remote API family and the signaling
// bridge, so the WebView never leaves its loopback origin.
const REMOTE_CLIENT_API = "/api/client-remote";
const REMOTE_PROXY_PREFIX = "/api/client-remote/proxy";
const REMOTE_ROUTE_PARAM = "remote";
const REMOTE_ROOM_PARAM = "room";
const CLIENT_PATH = "/client";
const ROOM_PATH_PREFIX = "/r/";
const ROOM_ID_PATTERN = /^[1-9][0-9]{3}$/;
const MAX_GRANT_LENGTH = 512;
const FINGERPRINT_PATTERN = /^[0-9A-F]{64}$/;

export const remoteInfoSchema = z.object({
  service: z.string().optional(),
  protocol: z.string().optional(),
  siteAccessRequired: z.boolean().default(false),
  roomCreation: z.boolean().default(false),
  apiOrigin: z.string().optional(),
  certificateSha256: z.string().optional(),
  capabilities: runtimeCapabilitiesSchema.optional(),
});
export type RemoteInfo = z.infer<typeof remoteInfoSchema>;

export const remoteProbeResultSchema = z.object({
  origin: z.string(),
  hostLabel: z.string(),
  tlsStatus: z.enum(["trusted", "untrusted"]),
  certificateSha256: z.string().optional(),
  info: remoteInfoSchema,
  roomId: z.string().optional(),
  grant: z.string().optional(),
});
export type RemoteProbeResult = z.infer<typeof remoteProbeResultSchema>;

export const remoteSessionStatusSchema = z.object({
  connected: z.boolean(),
  origin: z.string().optional(),
  hostLabel: z.string().optional(),
  certificateSha256: z.string().optional(),
  siteAccessRequired: z.boolean().default(false),
  roomCreation: z.boolean().default(false),
  protocol: z.string().optional(),
  capabilities: runtimeCapabilitiesSchema.optional(),
});
export type RemoteSessionStatus = z.infer<typeof remoteSessionStatusSchema>;

/** One host the user allowed, with the certificate it was allowed with. */
export const remoteHostSchema = z.object({
  origin: z.string(),
  fingerprint: z.string().optional(),
});
export type RemoteHost = z.infer<typeof remoteHostSchema>;

const remoteHostListSchema = z.array(remoteHostSchema).nullish().transform(hosts => hosts ?? []);
const remoteHostsSchema = z.object({ remoteHosts: remoteHostListSchema });

/** Which surface the launcher page shows while a remote session is active. */
export type RemoteViewRoute =
  | { kind: "launcher" }
  | { kind: "host" }
  | { kind: "join" }
  | { kind: "viewer"; roomId: string };

export type RemoteViewSurface = Exclude<RemoteViewRoute, { kind: "launcher" }>;

export interface RemoteEndpointState {
  /** The active remote session, or null while the App runs against its own origin. */
  session: RemoteSessionStatus | null;
  viewRoute: RemoteViewRoute;
}

/** One parsed remote address: where to connect and what the invitation settled. */
export interface RemoteDescriptor {
  /** The address exactly as given, which is what the launcher parses again. */
  target: string;
  origin?: string;
  roomId?: string;
  grant?: string;
  fingerprint?: string;
}

/**
 * Reads the three published address forms: a piik://connect invitation, an
 * HTTP(S) URL including a Browser invite, and a bare host.
 */
export function parseRemoteDescriptor(raw: string): RemoteDescriptor | null {
  const value = raw.trim();
  if (!value) return null;
  if (/^piik:\/\//i.test(value)) return parseInviteDescriptor(value);
  const parsed = parseURL(value.includes("://") ? value : `https://${value}`);
  if (!parsed) return null;
  const origin = originOf(parsed);
  if (!origin) return null;
  const roomId = roomFromPath(parsed.pathname);
  if (roomId === null) return null;
  const grant = grantFromFragment(parsed.hash);
  if (grant === null) return null;
  return {
    target: value,
    origin,
    ...(roomId ? { roomId } : {}),
    ...(grant ? { grant } : {}),
  };
}

/** Builds the piik://connect invitation the desktop App hands to another App. */
export function buildRemoteDescriptor({
  origin,
  roomId,
  grant,
  fingerprint,
}: {
  origin: string;
  roomId?: string;
  grant?: string;
  fingerprint?: string;
}): string {
  const params = new URLSearchParams({ origin: originFromValue(origin) ?? origin.trim() });
  if (roomId) params.set("room", roomId);
  if (grant) params.set("grant", grant);
  if (fingerprint) params.set("fp", fingerprint);
  return `piik://connect?${params.toString()}`;
}

// parseInviteDescriptor reads piik://connect?origin=&room=&grant=&fp=.
function parseInviteDescriptor(value: string): RemoteDescriptor | null {
  const parsed = parseURL(value);
  if (!parsed || parsed.username || parsed.password) return null;
  if (parsed.hostname.toLowerCase() !== "connect") return null;
  const origin = originFromValue(parsed.searchParams.get("origin") ?? "");
  if (!origin) return null;
  const roomId = (parsed.searchParams.get("room") ?? "").trim();
  if (roomId && !ROOM_ID_PATTERN.test(roomId)) return null;
  const grant = validateGrant(parsed.searchParams.get("grant") ?? "");
  if (grant === null) return null;
  const fingerprint = normalizeFingerprint(parsed.searchParams.get("fp") ?? "");
  if (fingerprint === null) return null;
  return {
    target: value,
    origin,
    ...(roomId ? { roomId } : {}),
    ...(grant ? { grant } : {}),
    ...(fingerprint ? { fingerprint } : {}),
  };
}

function parseURL(value: string): URL | null {
  try {
    return new URL(value);
  } catch {
    return null;
  }
}

function originFromValue(raw: string): string | null {
  const value = raw.trim();
  if (!value) return null;
  const parsed = parseURL(value.includes("://") ? value : `https://${value}`);
  return parsed ? originOf(parsed) : null;
}

// originOf reduces a parsed URL to the scheme and host a session uses.
function originOf(parsed: URL): string | null {
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
  if (parsed.username || parsed.password || !parsed.hostname) return null;
  return `${parsed.protocol}//${parsed.host.toLowerCase()}`;
}

// Returns the room code, "" when the path carries none, or null when invalid.
function roomFromPath(path: string): string | null {
  if (!path.startsWith(ROOM_PATH_PREFIX)) return "";
  const roomId = path.slice(ROOM_PATH_PREFIX.length);
  return ROOM_ID_PATTERN.test(roomId) ? roomId : null;
}

// Returns the grant, "" when the fragment carries none, or null when invalid.
function grantFromFragment(fragment: string): string | null {
  const value = fragment.replace(/^#/, "");
  if (!value) return "";
  const params = new URLSearchParams(value);
  return validateGrant(params.get("v") ?? params.get("grant") ?? "");
}

function validateGrant(grant: string): string | null {
  if (!grant) return "";
  if (grant.length > MAX_GRANT_LENGTH || /[ \t\r\n]/.test(grant)) return null;
  return grant;
}

// normalizeFingerprint mirrors the App: uppercase hex without separators.
function normalizeFingerprint(value: string): string | null {
  const fingerprint = value.trim().replace(/:/g, "").toUpperCase();
  if (!fingerprint) return "";
  return FINGERPRINT_PATTERN.test(fingerprint) ? fingerprint : null;
}

let endpointState: RemoteEndpointState = {
  session: null,
  viewRoute: readRemoteViewRoute(),
};
const endpointListeners = new Set<() => void>();
let popstateWatched = false;

export function getRemoteEndpoint(): RemoteEndpointState {
  return endpointState;
}

export function subscribeRemoteEndpoint(listener: () => void): () => void {
  endpointListeners.add(listener);
  watchPopState();
  return () => {
    endpointListeners.delete(listener);
  };
}

export function useRemoteEndpoint(): RemoteEndpointState {
  return useSyncExternalStore(subscribeRemoteEndpoint, getRemoteEndpoint, getRemoteEndpoint);
}

/** The proxied path while a remote session is active, otherwise the local one. */
export function apiPath(path: string): string {
  return endpointState.session ? `${REMOTE_PROXY_PREFIX}${path}` : path;
}

/** The signaling socket of whichever origin actually serves this page's rooms. */
export function signalWebSocketUrl(): string {
  const url = new URL(
    endpointState.session ? `${REMOTE_PROXY_PREFIX}/signal` : "/signal",
    window.location.href,
  );
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

/** The host the user is actually sharing with, for copy that names a site. */
export function siteHostLabel(): string {
  return endpointState.session?.hostLabel || window.location.host;
}

export async function probeRemoteHost(
  target: string,
  acceptedFingerprint?: string,
  signal?: AbortSignal,
): Promise<RemoteProbeResult> {
  const payload = {
    target,
    ...(acceptedFingerprint ? { acceptedFingerprint } : {}),
  };
  const body = isMyGo()
    ? await callRemoteIPC("Remote.Probe", [payload], signal)
    : await postRemoteJSON(`${REMOTE_CLIENT_API}/probe`, payload, signal);
  const parsed = remoteProbeResultSchema.safeParse(body);
  if (!parsed.success) throw new ApiError(say("host.err.serverError"), 502);
  return parsed.data;
}

export async function connectRemoteHost(
  request: { target: string; acceptedFingerprint?: string; sitePassword?: string },
  signal?: AbortSignal,
): Promise<RemoteSessionStatus> {
  const payload = {
    target: request.target,
    ...(request.acceptedFingerprint ? { acceptedFingerprint: request.acceptedFingerprint } : {}),
    ...(request.sitePassword ? { sitePassword: request.sitePassword } : {}),
  };
  const body = isMyGo()
    ? await callRemoteIPC("Remote.Connect", [payload], signal)
    : await postRemoteJSON(`${REMOTE_CLIENT_API}/connect`, payload, signal);
  const status = parseRemoteSessionStatus(body);
  setRemoteSession(status.connected ? status : null);
  return status;
}

export async function fetchRemoteStatus(signal?: AbortSignal): Promise<RemoteSessionStatus> {
  const body = isMyGo()
    ? await callRemoteIPC("Remote.Status", [], signal)
    : await getRemoteJSON(`${REMOTE_CLIENT_API}/status`, signal);
  const status = parseRemoteSessionStatus(body);
  setRemoteSession(status.connected ? status : null);
  // A session the App no longer holds cannot own a remote route.
  if (!status.connected) clearRemoteViewRoute();
  return status;
}

export async function disconnectRemoteHost(): Promise<void> {
  if (isMyGo()) {
    await callRemoteIPC("Remote.Disconnect", []);
  } else {
    await postRemoteJSON(`${REMOTE_CLIENT_API}/disconnect`, {});
  }
  setRemoteSession(null);
  clearRemoteViewRoute();
}

export async function setRemoteHostSaved(
  origin: string,
  fingerprint: string,
  allowed: boolean,
): Promise<RemoteHost[]> {
  if (isMyGo()) {
    const body = await callRemoteIPC("Remote.SetSavedHost", [{ origin, fingerprint, allowed }]);
    const parsed = remoteHostListSchema.safeParse(body);
    if (!parsed.success) throw new ApiError(say("host.err.serverError"), 502);
    return parsed.data;
  }
  const body = await postRemoteJSON(`${REMOTE_CLIENT_API}/host`, { origin, fingerprint, allowed });
  const parsed = remoteHostsSchema.safeParse(body);
  if (!parsed.success) throw new ApiError(say("host.err.serverError"), 502);
  return parsed.data.remoteHosts;
}

/** Shows a remote surface in place, keeping the window on the launcher origin. */
export function enterRemoteRoute(route: RemoteViewSurface, controlPort?: number): void {
  activateClient(controlPort);
  setViewRoute(route);
  writeRemoteViewRoute(route, "push");
}

export function returnToRemoteLauncher(): void {
  setViewRoute({ kind: "launcher" });
  writeRemoteViewRoute({ kind: "launcher" }, "replace");
}

function parseRemoteSessionStatus(body: unknown): RemoteSessionStatus {
  const parsed = remoteSessionStatusSchema.safeParse(body);
  if (!parsed.success) throw new ApiError(say("host.err.serverError"), 502);
  return parsed.data;
}

async function postRemoteJSON(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<unknown> {
  return remoteJSON(path, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, signal);
}

async function getRemoteJSON(path: string, signal?: AbortSignal): Promise<unknown> {
  return remoteJSON(path, { headers: { Accept: "application/json" }, cache: "no-store" }, signal);
}

async function remoteJSON(path: string, init: RequestInit, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { ...init, signal });
  if (!response.ok) throw new ApiError(remoteFailureMessage(response.status), response.status);
  try {
    return await response.json();
  } catch {
    throw new ApiError(say("host.err.serverError"), response.status);
  }
}

async function callRemoteIPC(method: string, args: unknown[], signal?: AbortSignal): Promise<unknown> {
  try {
    return signal ? await callWith(method, args, { signal }) : await call(method, ...args);
  } catch (err) {
    if (signal?.aborted || (err instanceof Error && err.name === "AbortError")) throw err;
    if (isCallError(err)) {
      const match = /^\[status=(\d+)\]/.exec(err.message);
      const status = match ? Number(match[1]) : 502;
      throw new ApiError(remoteFailureMessage(status), status);
    }
    throw new ApiError(say("host.err.serverError"), 502);
  }
}

function remoteFailureMessage(status: number): string {
  if (status === 401) return say("gate.wrong");
  if (status === 502) return say("gate.connectFailed");
  return say("gate.serviceUnavailable", { status: String(status) });
}

function setRemoteSession(session: RemoteSessionStatus | null): void {
  if (endpointState.session === session) return;
  updateEndpoint({ session, viewRoute: endpointState.viewRoute });
}

function setViewRoute(route: RemoteViewRoute): void {
  if (sameViewRoute(endpointState.viewRoute, route)) return;
  updateEndpoint({ session: endpointState.session, viewRoute: route });
}

function clearRemoteViewRoute(): void {
  if (endpointState.viewRoute.kind === "launcher") return;
  setViewRoute({ kind: "launcher" });
  writeRemoteViewRoute({ kind: "launcher" }, "replace");
}

function updateEndpoint(next: RemoteEndpointState): void {
  endpointState = next;
  for (const listener of endpointListeners) listener();
}

function sameViewRoute(left: RemoteViewRoute, right: RemoteViewRoute): boolean {
  if (left.kind !== right.kind) return false;
  return left.kind !== "viewer" || (right.kind === "viewer" && left.roomId === right.roomId);
}

// The URL is the launcher's route authority: a reload and the back button both
// land on the same surface the query names.
export function readRemoteViewRoute(): RemoteViewRoute {
  const url = pageURL();
  if (!url || !/^\/client\/?$/.test(url.pathname)) return { kind: "launcher" };
  const surface = url.searchParams.get(REMOTE_ROUTE_PARAM);
  if (surface === "host" || surface === "join") return { kind: surface };
  if (surface === "viewer") {
    const roomId = url.searchParams.get(REMOTE_ROOM_PARAM) ?? "";
    return isValidRoomId(roomId) ? { kind: "viewer", roomId } : { kind: "launcher" };
  }
  return { kind: "launcher" };
}

// Only the launcher's own two parameters change; every other one (the debug
// opt-in included) stays as the page received it.
function writeRemoteViewRoute(route: RemoteViewRoute, mode: "push" | "replace"): void {
  const url = pageURL();
  if (!url) return;
  url.pathname = CLIENT_PATH;
  url.searchParams.delete(REMOTE_ROUTE_PARAM);
  url.searchParams.delete(REMOTE_ROOM_PARAM);
  if (route.kind !== "launcher") url.searchParams.set(REMOTE_ROUTE_PARAM, route.kind);
  if (route.kind === "viewer") url.searchParams.set(REMOTE_ROOM_PARAM, route.roomId);
  // An invitation grant never belongs in the address bar.
  url.hash = "";
  const target = `${url.pathname}${url.search}`;
  if (mode === "push") window.history.pushState(window.history.state, "", target);
  else window.history.replaceState(window.history.state, "", target);
}

// The page URL, or null where no document owns one (a non-browser host).
function pageURL(): URL | null {
  if (typeof window === "undefined") return null;
  try {
    return new URL(window.location.href);
  } catch {
    return null;
  }
}

function watchPopState(): void {
  if (popstateWatched || typeof window === "undefined") return;
  popstateWatched = true;
  window.addEventListener("popstate", onPopState);
}

function onPopState(): void {
  setViewRoute(readRemoteViewRoute());
}
