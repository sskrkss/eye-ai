/**
 * Shared utilities for sign-in and sign-up pages.
 */
const authFetchOpts = { credentials: 'include', headers: { 'Content-Type': 'application/json' } };

// ========== PASSWORD SHOW/HIDE TOGGLE ==========
const EYE_OPEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

// Wrap every password input with a toggle that reveals/hides its value.
function initPasswordToggles() {
    document.querySelectorAll('input[type="password"]').forEach((input) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'password-wrapper';
        input.parentNode.insertBefore(wrapper, input);
        wrapper.appendChild(input);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'password-toggle';
        btn.innerHTML = EYE_OPEN;
        btn.setAttribute('aria-label', t('show_password'));
        wrapper.appendChild(btn);

        btn.addEventListener('click', () => {
            const reveal = input.type === 'password';
            input.type = reveal ? 'text' : 'password';
            btn.innerHTML = reveal ? EYE_OFF : EYE_OPEN;
            btn.setAttribute('aria-label', reveal ? t('hide_password') : t('show_password'));
        });
    });
}

document.addEventListener('DOMContentLoaded', initPasswordToggles);

function authApiPost(url, body) {
    return fetch(url, {
        ...authFetchOpts,
        method: 'POST',
        body: JSON.stringify(body)
    });
}

function showAuthMessage(text, type) {
    const result = document.getElementById('result');
    if (!result) return;
    const messageClass = {
        success: 'message-success',
        error: 'message-error',
        info: 'message-info'
    }[type] || 'message-info';
    result.innerHTML = `<p class="${messageClass}">${text}</p>`;
    setTimeout(() => {
        if (result.innerHTML.includes(text)) result.innerHTML = '';
    }, 5000);
}

function parseAuthError(result) {
    if (Array.isArray(result.detail)) {
        const first = result.detail[0];
        return first?.msg || t('request_failed');
    }
    if (result.detail && typeof result.detail === 'string') {
        return translateApiError(result.detail);
    }
    return t('request_failed');
}

async function submitAuthForm(options) {
    const {
        url,
        body,
        submitBtn,
        loadingText,
        doneText,
        successMessage,
        redirectUrl,
        redirectDelay = 1000,
        getErrorMsg = parseAuthError
    } = options;
    if (!submitBtn) return;
    submitBtn.disabled = true;
    submitBtn.innerText = loadingText;
    try {
        const response = await authApiPost(url, body);
        const result = await response.json();
        if (response.ok) {
            showAuthMessage(successMessage, 'success');
            if (redirectUrl) setTimeout(() => { window.location.href = redirectUrl; }, redirectDelay);
        } else {
            showAuthMessage('❌ ' + getErrorMsg(result), 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showAuthMessage('❌ ' + t('network_error'), 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = doneText;
    }
}
