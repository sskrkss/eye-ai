document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('signInForm');
    const submitBtn = document.getElementById('submitBtn');

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const login = document.getElementById('email_or_username').value.trim();
            const password = document.getElementById('plain_password').value;
            if (!login || !password) {
                showAuthMessage('❌ ' + t('fill_all_fields'), 'error');
                return;
            }
            submitAuthForm({
                url: '/api/sign-in',
                body: { email_or_username: login, plain_password: password },
                submitBtn,
                loadingText: t('signing_in'),
                doneText: t('signin_btn'),
                successMessage: '✅ ' + t('signed_in'),
                redirectUrl: '/',
                redirectDelay: 1000
            });
        });
    }
});