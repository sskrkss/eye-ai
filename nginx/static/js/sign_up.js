document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('signUpForm');
    const submitBtn = document.getElementById('submitBtn');
    const passwordInput = document.getElementById('plain_password');
    const confirmInput = document.getElementById('confirm_password');

    function checkPasswordMatch() {
        confirmInput.classList.remove('password-match', 'password-mismatch');
        if (confirmInput.value.length === 0) return;
        if (passwordInput.value === confirmInput.value) {
            confirmInput.classList.add('password-match');
        } else {
            confirmInput.classList.add('password-mismatch');
        }
    }

    passwordInput.addEventListener('input', checkPasswordMatch);
    confirmInput.addEventListener('input', checkPasswordMatch);

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const companyName = document.getElementById('company_name').value.trim();
            const lastName = document.getElementById('last_name').value.trim();
            const firstName = document.getElementById('first_name').value.trim();
            const middleName = document.getElementById('middle_name').value.trim();
            const email = document.getElementById('email').value.trim();
            const username = document.getElementById('username').value.trim();
            const password = passwordInput.value;
            const confirm = confirmInput.value;

            if (!companyName || !lastName || !firstName || !email || !username || !password || !confirm) {
                showAuthMessage('❌ ' + t('fill_all_fields'), 'error');
                return;
            }
            if (password.length < 8) {
                showAuthMessage('❌ ' + t('pw_too_short'), 'error');
                return;
            }
            if (password !== confirm) {
                showAuthMessage('❌ ' + t('pw_mismatch'), 'error');
                return;
            }

            const body = { company_name: companyName, last_name: lastName, first_name: firstName, email, username, plain_password: password };
            if (middleName) body.middle_name = middleName;

            submitAuthForm({
                url: '/api/sign-up',
                body,
                submitBtn,
                loadingText: t('creating_account'),
                doneText: t('create_account'),
                successMessage: '✅ ' + t('account_created'),
                redirectUrl: '/',
                redirectDelay: 2000,
                getErrorMsg: (result) => {
                    if (Array.isArray(result.detail)) return result.detail[0]?.msg || t('registration_failed');
                    if (typeof result.detail === 'string') return translateApiError(result.detail);
                    return t('registration_failed');
                }
            });
        });
    }
});