// Remote App connection: this App joins a Piik site running on another device.
// The launcher keeps the origin; the host is shown in place of the launcher.
import { useEffect, useId, useRef, useState, type FormEvent, type RefObject } from "react";
import {
  connectRemoteHost,
  disconnectRemoteHost,
  enterRemoteRoute,
  parseRemoteDescriptor,
  probeRemoteHost,
  setRemoteHostSaved,
  useRemoteEndpoint,
  type RemoteDescriptor,
  type RemoteHost,
  type RemoteProbeResult,
  type RemoteSessionStatus,
  type RemoteViewSurface,
} from "../../lib/remote-endpoint";
import { storeViewerGrant } from "../../lib/session";
import { ApiError } from "../../lib/api-error";
import { useCopy } from "../../ui/copy";
import { Glyph } from "../../ui/icons";
import { Btn, Pill } from "./primitives";

/** What the address form knows about the site it currently points at. */
export type RemoteConnectState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "invalid" }
  | { kind: "untrusted"; probe: RemoteProbeResult; descriptor: RemoteDescriptor }
  | { kind: "trusted"; probe: RemoteProbeResult; descriptor: RemoteDescriptor };

export interface RemoteConnectViewProps {
  session: RemoteSessionStatus | null;
  remoteHosts: readonly RemoteHost[];
  state: RemoteConnectState;
  address: string;
  passphrase: string;
  passphraseRef: RefObject<HTMLInputElement | null>;
  rememberHost: boolean;
  /** A rejected passphrase asks for the field even when the site hid the need. */
  needsPassphrase: boolean;
  error: string | null;
  busy: boolean;
  onAddressChange: (value: string) => void;
  onPassphraseChange: (value: string) => void;
  onRememberHostChange: (value: boolean) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onTrust: () => void;
  onCancel: () => void;
  onOpen: (route: RemoteViewSurface) => void;
  onConnect: (route: RemoteViewSurface) => void;
  onConnectSaved: (host: RemoteHost) => void;
  onRemoveHost: (host: RemoteHost) => void;
  onDisconnect: () => void;
}

/** Presentation only: every state is reachable without a live endpoint. */
export function RemoteConnectView({
  session,
  remoteHosts,
  state,
  address,
  passphrase,
  passphraseRef,
  rememberHost,
  needsPassphrase,
  error,
  busy,
  onAddressChange,
  onPassphraseChange,
  onRememberHostChange,
  onSubmit,
  onTrust,
  onCancel,
  onOpen,
  onConnect,
  onConnectSaved,
  onRemoveHost,
  onDisconnect,
}: RemoteConnectViewProps) {
  const { vis, t } = useCopy();
  const id = useId();
  const hidden = vis ? "visually-hidden" : undefined;
  const passphraseHintId = `${id}-passphrase`;

  if (session) {
    const label = session.hostLabel || session.origin || "";
    return (
      <section className="lr-client-remote is-connected" aria-label={t("client.remote.sessionLabel")}>
        <div className="lr-client-remote-host-line">
          <Glyph name="globe" size={20} />
          <strong dir="ltr">{label}</strong>
          <span className="lr-client-remote-state">
            <Glyph name="check" size={16} />
            <span className={hidden}>{t("client.remote.connected")}</span>
          </span>
        </div>
        <div className="lr-client-remote-actions">
          {session.roomCreation && <Btn icon="door" title="client.remote.createRoom" cap="client.remote.createRoom"
            size="compact" disabled={busy} onClick={() => onOpen({ kind: "host" })} />}
          <Btn icon="arrowRight" title="client.remote.join" cap="client.remote.join"
            size="compact" disabled={busy} onClick={() => onOpen({ kind: "join" })} />
          <Btn icon="x" title="client.remote.disconnect" cap="client.remote.disconnect" size="compact" quiet
            disabled={busy} hint="hint-close" onClick={onDisconnect} />
        </div>
      </section>
    );
  }

  const roomId = state.kind === "trusted" ? state.descriptor.roomId ?? null : null;
  return (
    <section className="lr-client-remote" aria-label={t("client.remote.title")}>
      <div className="lr-client-remote-host-line">
        <Glyph name="globe" size={20} />
        <strong className={hidden}>{t("client.remote.title")}</strong>
      </div>
      <p className={hidden}>{t("client.remote.hint")}</p>

      {remoteHosts.length > 0 && <div className="lr-client-remote-saved" role="list" aria-label={t("client.remote.savedHosts")}>
        {remoteHosts.map(host => <div key={host.origin} className="lr-client-remote-saved-item" role="listitem">
          <button type="button" className="lr-btn is-dense is-quiet lr-client-remote-saved-connect" disabled={busy}
            onClick={() => onConnectSaved(host)}>
            <Glyph name={host.fingerprint ? "lock" : "globe"} size={16} />
            <span dir="ltr">{host.origin}</span>
          </button>
          <Btn icon="x" title="client.remote.removeHost" size="dense" quiet disabled={busy}
            hint="hint-close" onClick={() => onRemoveHost(host)} />
        </div>)}
      </div>}

      <form className="lr-client-remote-form" onSubmit={onSubmit}>
        <label className="lr-input">
          <Glyph name="link" size={18} />
          <input
            type="text"
            value={address}
            autoComplete="off"
            spellCheck={false}
            dir="ltr"
            placeholder={vis ? "" : t("client.remote.addressPlaceholder")}
            aria-label={t("client.remote.address")}
            onChange={event => onAddressChange(event.target.value)}
          />
        </label>
        <Btn icon="arrowRight" title="client.remote.connect" cap="client.remote.connect" type="submit" tone="primary"
          size="compact" busy={state.kind === "checking"} disabled={busy || !address.trim()} />
      </form>

      {state.kind === "checking" && <p role="status" className={hidden}>{t("client.remote.checking")}</p>}
      {state.kind === "invalid" && <Pill icon="alert" tone="bad" label={t("client.remote.invalidAddress")} alert comic="signal-failed" />}
      {error && <Pill icon="alert" tone="bad" label={error} alert comic="access-denied" />}

      {state.kind === "untrusted" && <div className="lr-client-remote-trust">
        <strong className={hidden}>{t("client.remote.untrustedTitle")}</strong>
        <p className={hidden}>{t("client.remote.untrustedHint")}</p>
        <div className="lr-client-remote-fp">
          <span className={hidden}>{t("client.remote.fingerprint")}</span>
          <code dir="ltr">{state.probe.certificateSha256 ?? ""}</code>
        </div>
        <div className="lr-client-remote-actions">
          <Btn icon="lock" title="client.remote.trust" cap="client.remote.trust" tone="primary" size="compact"
            disabled={busy} hint="warning" onClick={onTrust} />
          <Btn icon="x" title="common.cancel" cap="common.cancel" size="compact" quiet
            disabled={busy} hint="hint-close" onClick={onCancel} />
        </div>
      </div>}

      {state.kind === "trusted" && <div className="lr-client-remote-site">
        <div className="lr-client-remote-host-line">
          <Glyph name="check" size={18} className="lr-client-remote-check" />
          <strong dir="ltr">{state.probe.hostLabel}</strong>
        </div>
        <label className="lr-client-remote-remember">
          <input type="checkbox" checked={rememberHost} disabled={busy}
            aria-label={t("client.remote.saveHost")}
            onChange={event => onRememberHostChange(event.target.checked)} />
          <Glyph name="lock" size={16} />
          <span className={hidden}>{t("client.remote.saveHost")}</span>
        </label>
        {(state.probe.info.siteAccessRequired || needsPassphrase) && <>
          <label className="lr-input">
            <Glyph name="key" size={18} />
            <input
              ref={passphraseRef}
              type="password"
              value={passphrase}
              autoComplete="off"
              spellCheck={false}
              placeholder={vis ? "" : t("client.remote.passphrase")}
              aria-label={t("client.remote.passphrase")}
              aria-describedby={passphraseHintId}
              onChange={event => onPassphraseChange(event.target.value)}
            />
          </label>
          <p id={passphraseHintId} className={hidden}>{t("client.remote.passphraseHint")}</p>
        </>}
        <div className="lr-client-remote-actions">
          {roomId ? <button type="button" className="lr-btn is-primary is-compact" disabled={busy}
            aria-label={t("client.remote.joinRoom", { room: roomId })}
            onClick={() => onConnect({ kind: "viewer", roomId })}>
            <Glyph name="arrowRight" size={19} />
            {vis ? null : <span className="lr-cap">{t("client.remote.joinRoom", { room: roomId })}</span>}
          </button> : <>
            <Btn icon="arrowRight" title="client.remote.join" cap="client.remote.join" tone="primary" size="compact"
              disabled={busy} onClick={() => onConnect({ kind: "join" })} />
            {state.probe.info.roomCreation && <Btn icon="door" title="client.remote.createRoom" cap="client.remote.createRoom"
              size="compact" disabled={busy} onClick={() => onConnect({ kind: "host" })} />}
          </>}
        </div>
      </div>}
    </section>
  );
}

/** The launcher's remote entry: probes a site, then owns the session it opens. */
export function RemoteConnectCard({
  remoteHosts,
  onRemoteHostsChange,
  controlPort,
}: {
  remoteHosts: readonly RemoteHost[];
  onRemoteHostsChange: (hosts: RemoteHost[]) => void;
  controlPort?: number;
}) {
  const { session } = useRemoteEndpoint();
  const { t } = useCopy();
  const [state, setState] = useState<RemoteConnectState>({ kind: "idle" });
  const [address, setAddress] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [rememberHost, setRememberHost] = useState(false);
  const [needsPassphrase, setNeedsPassphrase] = useState(false);
  const [passphraseFocus, setPassphraseFocus] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const passphraseRef = useRef<HTMLInputElement>(null);
  // The field appears with the rejection, so focus follows the rendered input.
  useEffect(() => {
    if (passphraseFocus) passphraseRef.current?.focus();
  }, [passphraseFocus]);

  function failure(cause: unknown): void {
    if (cause instanceof ApiError && cause.status === 401) {
      setError(t("gate.wrong"));
      setNeedsPassphrase(true);
      setPassphraseFocus(count => count + 1);
      return;
    }
    setError(cause instanceof ApiError ? cause.message : t("gate.connectFailed"));
  }

  // A password draft never outlives the site it was typed for.
  function reset(): void {
    setState({ kind: "idle" });
    setPassphrase("");
    setRememberHost(false);
    setNeedsPassphrase(false);
    setError(null);
  }

  async function connectTo(route: RemoteViewSurface, result: RemoteProbeResult, descriptor: RemoteDescriptor): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await connectRemoteHost({
        target: descriptor.target,
        ...(result.certificateSha256 ? { acceptedFingerprint: result.certificateSha256 } : {}),
        ...(passphrase ? { sitePassword: passphrase } : {}),
      });
      if (rememberHost) {
        await setRemoteHostSaved(result.origin, result.certificateSha256 ?? "", true)
          .then(onRemoteHostsChange, () => undefined);
      }
      if (descriptor.roomId && descriptor.grant) storeViewerGrant(descriptor.roomId, descriptor.grant);
      setPassphrase("");
      enterRemoteRoute(route, controlPort);
    } catch (cause) {
      failure(cause);
    } finally {
      setBusy(false);
    }
  }

  // Probing settles the certificate before any credential leaves this page.
  async function probe(descriptor: RemoteDescriptor, acceptedFingerprint?: string, connectNow = false): Promise<void> {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await probeRemoteHost(descriptor.target, acceptedFingerprint);
      const resolved = { ...descriptor, origin: result.origin };
      if (result.tlsStatus !== "trusted") {
        setState({ kind: "untrusted", probe: result, descriptor: resolved });
        return;
      }
      if (!connectNow || result.info.siteAccessRequired) {
        setState({ kind: "trusted", probe: result, descriptor: resolved });
        return;
      }
      await connectTo({ kind: "join" }, result, resolved);
    } catch (cause) {
      failure(cause);
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (busy) return;
    const descriptor = parseRemoteDescriptor(address);
    if (!descriptor) {
      setState({ kind: "invalid" });
      setError(null);
      return;
    }
    setState({ kind: "checking" });
    void probe(descriptor);
  }

  function trust(): void {
    if (state.kind !== "untrusted" || busy) return;
    setRememberHost(true);
    setState({ kind: "checking" });
    void probe(state.descriptor, state.probe.certificateSha256);
  }

  function connect(route: RemoteViewSurface): void {
    if (state.kind !== "trusted" || busy) return;
    void connectTo(route, state.probe, state.descriptor);
  }

  function connectSaved(host: RemoteHost): void {
    void probe({ target: host.origin, origin: host.origin }, host.fingerprint, true);
  }

  function removeHost(host: RemoteHost): void {
    if (busy) return;
    setBusy(true);
    void setRemoteHostSaved(host.origin, host.fingerprint ?? "", false)
      .then(onRemoteHostsChange, failure)
      .finally(() => setBusy(false));
  }

  function disconnect(): void {
    if (busy) return;
    setBusy(true);
    void disconnectRemoteHost().catch(failure).finally(() => setBusy(false));
  }

  return <RemoteConnectView
    session={session}
    remoteHosts={remoteHosts}
    state={state}
    address={address}
    passphrase={passphrase}
    passphraseRef={passphraseRef}
    rememberHost={rememberHost}
    needsPassphrase={needsPassphrase}
    error={error}
    busy={busy}
    onAddressChange={setAddress}
    onPassphraseChange={setPassphrase}
    onRememberHostChange={setRememberHost}
    onSubmit={submit}
    onTrust={trust}
    onCancel={reset}
    onOpen={route => enterRemoteRoute(route, controlPort)}
    onConnect={connect}
    onConnectSaved={connectSaved}
    onRemoveHost={removeHost}
    onDisconnect={disconnect}
  />;
}
