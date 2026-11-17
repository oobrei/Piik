import { say } from "../ui/copy";
import {
  apiPath,
  remoteInfoSchema,
  type RemoteInfo,
} from "./remote-endpoint";
import {
  createRoomResponseSchema,
  roomAccessUpdateResponseSchema,
  runtimeCapabilitiesSchema,
  type CreateRoomResponse,
  type CodeEntryPolicy,
  type RoomAccessUpdateRequest,
  type RoomAccessUpdateResponse,
  type RuntimeCapabilities,
} from "../../shared/protocol";

export interface SiteAccessStatus {
  required: boolean;
  authenticated: boolean;
}

import { ApiError } from "./api-error";
export { ApiError };

export async function responseBody(response: Response): Promise<unknown> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(say("host.err.serverError"), response.status);
  }
  if (body && typeof body === "object" && "error" in body) {
    if (response.status === 403 && body.error === "Origin not allowed") {
      throw new ApiError(say("api.originNotAllowed"), response.status);
    }
    if (response.status === 503 && body.error === "Room capacity reached") {
      throw new ApiError(say("host.err.roomCapacity"), response.status);
    }
  }
  return body;
}

function parseSiteAccessStatus(value: unknown): SiteAccessStatus {
  if (!value || typeof value !== "object") {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record.required !== "boolean" ||
    typeof record.authenticated !== "boolean"
  ) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return {
    required: record.required,
    authenticated: record.authenticated,
  };
}

export async function getSiteAccess(signal?: AbortSignal): Promise<SiteAccessStatus> {
  const response = await fetch(apiPath("/api/site-access"), {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      response.status === 401
        ? say("gate.expired")
        : say("gate.serviceUnavailable", {
            status: String(response.status),
          }),
      response.status,
    );
  }
  return parseSiteAccessStatus(body);
}

export async function getRuntimeCapabilities(signal?: AbortSignal): Promise<RuntimeCapabilities> {
  const response = await fetch(apiPath("/api/capabilities"), {
    headers: { Accept: "application/json" },
    signal,
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      say("gate.serviceUnavailable", { status: String(response.status) }),
      response.status,
    );
  }
  const parsed = runtimeCapabilitiesSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return parsed.data;
}

/** The host's own description, reachable before any remote session exists. */
export async function getRemoteInfo(signal?: AbortSignal): Promise<RemoteInfo> {
  const response = await fetch(apiPath("/api/remote/info"), {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      say("gate.serviceUnavailable", { status: String(response.status) }),
      response.status,
    );
  }
  const parsed = remoteInfoSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return parsed.data;
}

export async function authenticateSiteAccess(
  password: string,
): Promise<SiteAccessStatus> {
  const response = await fetch(apiPath("/api/site-access"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ password }),
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      response.status === 401
        ? say("gate.wrong")
        : say("gate.serviceUnavailable", {
            status: String(response.status),
          }),
      response.status,
    );
  }
  return parseSiteAccessStatus(body);
}

export async function createRoom(
  codeEntryPolicy: CodeEntryPolicy,
  roomPassword: string | null,
  preferredRoomId: string | null = null,
): Promise<CreateRoomResponse> {
  const response = await fetch(apiPath("/api/rooms"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      codeEntryPolicy,
      ...(roomPassword === null ? {} : { roomPassword }),
      ...(preferredRoomId === null ? {} : { preferredRoomId }),
    }),
  });

  const body = await responseBody(response);

  if (!response.ok) {
    throw new ApiError(
      response.status === 401
        ? say("gate.expired")
        : say("host.err.createRoomStatus", {
            status: String(response.status),
          }),
      response.status,
    );
  }

  const parsed = createRoomResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return parsed.data;
}

export async function replaceOwnedRoom(
  roomId: string,
  hostToken: string,
  codeEntryPolicy: CodeEntryPolicy,
  roomPassword: string | null,
): Promise<CreateRoomResponse> {
  const response = await fetch(apiPath(`/api/rooms/${roomId}/replacement`), {
    method: "POST",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${hostToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      codeEntryPolicy,
      ...(roomPassword === null ? {} : { roomPassword }),
    }),
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      response.status === 404
        ? say("viewer.msg.notFound")
        : say("host.err.replaceRoomStatus", {
            status: String(response.status),
          }),
      response.status,
    );
  }
  const parsed = createRoomResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return parsed.data;
}

export async function updateRoomAccess(
  roomId: string,
  hostToken: string,
  request: RoomAccessUpdateRequest,
): Promise<RoomAccessUpdateResponse> {
  const response = await fetch(apiPath(`/api/rooms/${roomId}/access`), {
    method: "POST",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${hostToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  const body = await responseBody(response);
  if (!response.ok) {
    throw new ApiError(
      response.status === 401
        ? say("gate.expired")
        : response.status === 404
          ? say("viewer.msg.notFound")
          : say("host.err.updateRoomStatus", {
              status: String(response.status),
            }),
      response.status,
    );
  }

  const parsed = roomAccessUpdateResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(say("host.err.serverError"), 502);
  }
  return parsed.data;
}
