// apps/Rubick/Sources/Backend/HotkeyService.swift
import AppKit
import Carbon.HIToolbox

/// Carbon-backed global hotkey with double-press detection.
///
/// Registers ⌥+Space via `RegisterEventHotKey`. On each press:
/// - If no prior press within `doublePressWindow` → start a timer.
///   After the window expires, fire `onSinglePress`.
/// - If a prior press is pending → cancel the timer, fire `onDoublePress`.
///
/// This gives a 300ms latency on single-press (the trade-off for
/// double-press detection), but Pulsar's entrance animation masks it.
@MainActor
final class HotkeyService {
    typealias Action = @MainActor () -> Void

    private var onSinglePress: Action?
    private var onDoublePress: Action?

    /// Window in seconds to detect a double-press.
    private let doublePressWindow: TimeInterval = 0.3

    /// Pending single-press timer. Non-nil means we're waiting to see
    /// if a second press arrives.
    private var pendingTimer: DispatchWorkItem?

    nonisolated(unsafe) private var hotKeyRef: EventHotKeyRef?
    nonisolated(unsafe) private var eventHandler: EventHandlerRef?

    /// Install the ⌥+Space hotkey with single/double press callbacks.
    ///
    /// - Parameters:
    ///   - onSinglePress: Fires after `doublePressWindow` with no second press.
    ///   - onDoublePress: Fires immediately on second press within window.
    /// - Returns: `false` if registration failed (another app owns ⌥+Space).
    @discardableResult
    func install(
        keyCode: UInt32 = UInt32(kVK_Space),
        modifierFlags: UInt32 = UInt32(optionKey),
        onSinglePress: @escaping Action,
        onDoublePress: @escaping Action
    ) -> Bool {
        uninstall()
        self.onSinglePress = onSinglePress
        self.onDoublePress = onDoublePress

        let hotKeyID = EventHotKeyID(signature: 0x5255_424B, id: 1)

        var hkRef: EventHotKeyRef?
        let status = RegisterEventHotKey(
            keyCode,
            modifierFlags,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hkRef
        )
        guard status == noErr, let hk = hkRef else {
            FileHandle.standardError.write(
                Data("[HotkeyService] RegisterEventHotKey failed (status=\(status))\n".utf8)
            )
            return false
        }
        hotKeyRef = hk

        var eventSpec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        let userPtr = Unmanaged.passUnretained(self).toOpaque()
        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            Self.handlerProc,
            1,
            &eventSpec,
            userPtr,
            &eventHandler
        )
        if installStatus != noErr {
            FileHandle.standardError.write(
                Data("[HotkeyService] InstallEventHandler failed (status=\(installStatus))\n".utf8)
            )
            UnregisterEventHotKey(hk)
            hotKeyRef = nil
            return false
        }
        return true
    }

    func uninstall() {
        pendingTimer?.cancel()
        pendingTimer = nil
        if let hk = hotKeyRef {
            UnregisterEventHotKey(hk)
            hotKeyRef = nil
        }
        if let h = eventHandler {
            RemoveEventHandler(h)
            eventHandler = nil
        }
    }

    deinit {
        if let hk = hotKeyRef { UnregisterEventHotKey(hk) }
        if let h = eventHandler { RemoveEventHandler(h) }
    }

    // MARK: - Double-press state machine

    private func handlePress() {
        if pendingTimer != nil {
            // Second press within window → double
            pendingTimer?.cancel()
            pendingTimer = nil
            onDoublePress?()
        } else {
            // First press → start timer
            let work = DispatchWorkItem { [weak self] in
                MainActor.assumeIsolated {
                    self?.pendingTimer = nil
                    self?.onSinglePress?()
                }
            }
            pendingTimer = work
            DispatchQueue.main.asyncAfter(
                deadline: .now() + doublePressWindow,
                execute: work
            )
        }
    }

    // MARK: - Carbon handler

    private static let handlerProc: EventHandlerProcPtr = {
        _, _, userData in
        guard let userData else { return OSStatus(eventNotHandledErr) }
        let svc = Unmanaged<HotkeyService>.fromOpaque(userData).takeUnretainedValue()
        MainActor.assumeIsolated {
            svc.handlePress()
        }
        return noErr
    }
}
