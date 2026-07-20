// ========== i18n ==========
// Shared translation layer for all pages. Default language: Russian.
// Loaded before every page script (see base.html), so `t()` and friends
// are available globally at render time.

const I18N = {
    ru: {
        // common
        logout: 'Выйти',
        loading: 'Загрузка...',
        something_wrong: 'Что-то пошло не так',
        try_again: 'Повторить',

        // diagnosis labels
        diag_unknown: 'Неизвестно',
        diag_no_dr: 'Нет ДР',
        diag_mild_dr: 'Легкая ДР',
        diag_moderate_dr: 'Умеренная ДР',
        diag_severe_dr: 'Тяжелая ДР',
        diag_proliferative_dr: 'Пролиферативная ДР',

        // gender
        gender_male: 'Мужской',
        gender_female: 'Женский',

        // task status
        status_processing: 'Обработка',
        status_completed: 'Завершено',
        status_failed: 'Ошибка',
        status_reviewed: 'Проверено',

        // dashboard
        patients_title: 'Пациенты',
        add_patient: 'Добавить пациента',
        no_patients: 'Пациентов пока нет. Добавьте первого, чтобы начать.',
        no_patients_filtered: 'Нет пациентов с выбранным диагнозом.',
        filter_diagnosis: 'Диагноз',
        filter_all: 'Все диагнозы',
        th_name: 'ФИО',
        th_age: 'Возраст',
        th_gender: 'Пол',
        th_diagnosis: 'Диагноз',

        // company card / scan counter
        license_demo: 'Демо',
        license_active: 'Активна',
        license_expired: 'Истекла',
        license_until: 'до',
        scans_label: 'Сканы',
        scans_remaining: 'Осталось',

        // licenses (super admin)
        licenses_title: 'Лицензии',
        th_company: 'Компания',
        th_expiry: 'Срок',
        max_scans_label: 'Лимит сканов',
        notes_label: 'Заметки',

        // doctors management
        doctors_title: 'Врачи',
        th_role: 'Роль',
        add_doctor: 'Добавить врача',
        add_doctor_btn: 'Добавить',
        role_admin: 'Администратор',
        role_doctor: 'Врач',
        promote: 'Сделать админом',
        demote: 'Снять админа',
        delete: 'Удалить',
        you: 'вы',
        confirm_delete_doctor: 'Удалить этого пользователя?',

        // add/edit patient modal
        modal_add_title: 'Добавить пациента',
        modal_edit_title: 'Редактировать пациента',
        f_last_name: 'Фамилия',
        f_first_name: 'Имя',
        f_middle_name: 'Отчество',
        optional: '(необязательно)',
        f_age: 'Возраст',
        f_gender: 'Пол',
        select_placeholder: 'Выберите...',
        cancel: 'Отмена',
        btn_add_patient: 'Добавить',
        adding: 'Добавление...',
        fill_required: 'Заполните все обязательные поля',
        failed_add_patient: 'Не удалось добавить пациента',

        // patient page
        back_patients: '← Пациенты',
        edit: 'Редактировать',
        years_old: 'лет',
        diagnosis_prefix: 'Диагноз:',
        save: 'Сохранить',
        saving: 'Сохранение...',
        failed_save: 'Не удалось сохранить',

        // upload card
        new_scan: 'Новый снимок глазного дна',
        badge_formats: '🖼️ JPEG · PNG · TIFF',
        badge_size: '📁 До 10 МБ',
        drop_here: 'Перетащите снимок сюда или <u>выберите</u>',
        drop_hint: 'JPEG, PNG, TIFF — до 10 МБ',
        analyze: 'Анализировать',
        click_to_change: 'нажмите, чтобы изменить',
        analyzing: 'Анализ изображения...',
        failed_start: 'Не удалось запустить анализ',
        network_error: 'Ошибка сети',
        analysis_failed: 'Анализ не выполнен',
        timeout: 'Время ожидания истекло — попробуйте снова',

        // screening history
        screening_history: 'История скринингов',
        no_screenings: 'Скринингов пока нет. Загрузите снимок, чтобы начать.',
        th_date: 'Дата',
        th_file: 'Файл',
        th_ai: 'Прогноз ИИ',
        th_doctor: 'Заключение врача',
        th_status: 'Статус',
        review_btn: 'Проверить',
        edit_btn: 'Редактировать',

        // review modal
        modal_review_title: 'Заключение врача',
        diagnosis: 'Диагноз',

        // auth (common)
        fill_all_fields: 'Заполните все поля',
        request_failed: 'Не удалось выполнить запрос',

        // sign in
        signin_btn: 'Войти',
        signing_in: 'Вход...',
        signed_in: 'Вход выполнен',
        email_or_username: 'Email или имя пользователя',
        password: 'Пароль',
        show_password: 'Показать пароль',
        hide_password: 'Скрыть пароль',
        no_account: 'Нет аккаунта?',
        create_one: 'Создать',
        ph_login: 'your@email.com или имя пользователя',

        // sign up
        company_name: 'Название организации',
        email: 'Email',
        username: 'Имя пользователя',
        confirm_password: 'Подтвердите пароль',
        create_account: 'Создать аккаунт',
        creating_account: 'Создание аккаунта...',
        account_created: 'Аккаунт успешно создан',
        have_account: 'Уже есть аккаунт?',
        pw_too_short: 'Пароль должен быть не менее 8 символов',
        pw_mismatch: 'Пароли не совпадают',
        registration_failed: 'Ошибка регистрации',
        ph_company: 'Городская больница №1',
    },
    en: {
        // common
        logout: 'Logout',
        loading: 'Loading...',
        something_wrong: 'Something went wrong',
        try_again: 'Try again',

        // diagnosis labels
        diag_unknown: 'Unknown',
        diag_no_dr: 'No DR',
        diag_mild_dr: 'Mild DR',
        diag_moderate_dr: 'Moderate DR',
        diag_severe_dr: 'Severe DR',
        diag_proliferative_dr: 'Proliferative DR',

        // gender
        gender_male: 'Male',
        gender_female: 'Female',

        // task status
        status_processing: 'Processing',
        status_completed: 'Completed',
        status_failed: 'Failed',
        status_reviewed: 'Reviewed',

        // dashboard
        patients_title: 'Patients',
        add_patient: 'Add patient',
        no_patients: 'No patients yet. Add your first patient to get started.',
        no_patients_filtered: 'No patients match the selected diagnosis.',
        filter_diagnosis: 'Diagnosis',
        filter_all: 'All diagnoses',
        th_name: 'Name',
        th_age: 'Age',
        th_gender: 'Gender',
        th_diagnosis: 'Diagnosis',

        // company card / scan counter
        license_demo: 'Demo',
        license_active: 'Active',
        license_expired: 'Expired',
        license_until: 'until',
        scans_label: 'Scans',
        scans_remaining: 'Remaining',

        // licenses (super admin)
        licenses_title: 'Licenses',
        th_company: 'Company',
        th_expiry: 'Valid until',
        max_scans_label: 'Scan limit',
        notes_label: 'Notes',

        // doctors management
        doctors_title: 'Doctors',
        th_role: 'Role',
        add_doctor: 'Add doctor',
        add_doctor_btn: 'Add doctor',
        role_admin: 'Admin',
        role_doctor: 'Doctor',
        promote: 'Make admin',
        demote: 'Remove admin',
        delete: 'Delete',
        you: 'you',
        confirm_delete_doctor: 'Delete this user?',

        // add/edit patient modal
        modal_add_title: 'Add patient',
        modal_edit_title: 'Edit patient',
        f_last_name: 'Last name',
        f_first_name: 'First name',
        f_middle_name: 'Middle name',
        optional: '(optional)',
        f_age: 'Age',
        f_gender: 'Gender',
        select_placeholder: 'Select...',
        cancel: 'Cancel',
        btn_add_patient: 'Add patient',
        adding: 'Adding...',
        fill_required: 'Please fill in all required fields',
        failed_add_patient: 'Failed to add patient',

        // patient page
        back_patients: '← Patients',
        edit: 'Edit',
        years_old: 'y.o.',
        diagnosis_prefix: 'Diagnosis:',
        save: 'Save',
        saving: 'Saving...',
        failed_save: 'Failed to save',

        // upload card
        new_scan: 'New retinal scan',
        badge_formats: '🖼️ JPEG · PNG · TIFF',
        badge_size: '📁 Max 10 MB',
        drop_here: 'Drop image here or <u>browse</u>',
        drop_hint: 'JPEG, PNG, TIFF — up to 10 MB',
        analyze: 'Analyze',
        click_to_change: 'click to change',
        analyzing: 'Analyzing image...',
        failed_start: 'Failed to start analysis',
        network_error: 'Network error',
        analysis_failed: 'Analysis failed',
        timeout: 'Timeout — please try again',

        // screening history
        screening_history: 'Screening history',
        no_screenings: 'No screenings yet. Upload a retinal image to get started.',
        th_date: 'Date',
        th_file: 'File',
        th_ai: 'AI Prediction',
        th_doctor: 'Doctor conclusion',
        th_status: 'Status',
        review_btn: 'Review',
        edit_btn: 'Edit',

        // review modal
        modal_review_title: "Doctor's conclusion",
        diagnosis: 'Diagnosis',

        // auth (common)
        fill_all_fields: 'Please fill in all fields',
        request_failed: 'Request failed',

        // sign in
        signin_btn: 'Sign in',
        signing_in: 'Signing in...',
        signed_in: 'Signed in successfully',
        email_or_username: 'Email or username',
        password: 'Password',
        show_password: 'Show password',
        hide_password: 'Hide password',
        no_account: 'No account?',
        create_one: 'Create one',
        ph_login: 'your@email.com or username',

        // sign up
        company_name: 'Company name',
        email: 'Email',
        username: 'Username',
        confirm_password: 'Confirm password',
        create_account: 'Create account',
        creating_account: 'Creating account...',
        account_created: 'Account successfully created',
        have_account: 'Already have an account?',
        pw_too_short: 'Password must be at least 8 characters',
        pw_mismatch: 'Passwords do not match',
        registration_failed: 'Registration failed',
        ph_company: 'City Hospital №1',
    },
};

// Backend (FastAPI) error `detail` strings are emitted in English. This maps the
// known ones to the current language; unknown details fall through unchanged.
const API_ERRORS = {
    'Cannot delete yourself': { ru: 'Нельзя удалить самого себя', en: 'Cannot delete yourself' },
    'Cannot change your own role': { ru: 'Нельзя изменить свою собственную роль', en: 'Cannot change your own role' },
    'Company with this name already exists': { ru: 'Организация с таким названием уже существует', en: 'Company with this name already exists' },
    'Company not found': { ru: 'Организация не найдена', en: 'Company not found' },
    'Image size exceeds 10 MB limit': { ru: 'Размер изображения превышает 10 МБ', en: 'Image size exceeds 10 MB limit' },
    'Invalid credentials': { ru: 'Неверный логин или пароль', en: 'Invalid email/username or password' },
    'License has expired': { ru: 'Срок действия лицензии истек', en: 'License has expired' },
    'License not found': { ru: 'Лицензия не найдена', en: 'License not found' },
    'Ml task is not permitted': { ru: 'Нет доступа к задаче', en: 'Task is not permitted' },
    'Ml task not found': { ru: 'Задача не найдена', en: 'Task not found' },
    'No active license': { ru: 'Нет активной лицензии', en: 'No active license' },
    'Only completed tasks can be reviewed': { ru: 'Заключение можно дать только по завершенной задаче', en: 'Only completed tasks can be reviewed' },
    'Patient not found': { ru: 'Пациент не найден', en: 'Patient not found' },
    'User not found': { ru: 'Пользователь не найден', en: 'User not found' },
    'User with this email or username already exists': { ru: 'Пользователь с таким email или именем уже существует', en: 'User with this email or username already exists' },
};

const SUPPORTED_LANGS = ['ru', 'en'];

function getLang() {
    const stored = localStorage.getItem('lang');
    return SUPPORTED_LANGS.includes(stored) ? stored : 'ru';
}

function setLang(lang) {
    if (!SUPPORTED_LANGS.includes(lang)) return;
    localStorage.setItem('lang', lang);
    document.documentElement.lang = lang;
}

function t(key) {
    const lang = getLang();
    return (I18N[lang] && I18N[lang][key]) ?? (I18N.en[key] ?? key);
}

// Badge color class by diagnosis severity. Shared by the dashboard and patient pages.
function diagnosisBadgeClass(d) {
    switch (d) {
        case 'no_dr': return 'badge-success';                  // green
        case 'mild_dr':
        case 'moderate_dr': return 'badge-yellow';             // yellow
        case 'severe_dr': return 'badge-warning';              // orange
        case 'proliferative_dr': return 'badge-danger';        // red
        default: return 'badge-neutral';                       // unknown / none
    }
}

// Translate a backend error `detail` string. Unknown strings pass through as-is.
function translateApiError(detail) {
    if (!detail || typeof detail !== 'string') return detail;
    const entry = API_ERRORS[detail];
    if (!entry) return detail;
    return entry[getLang()] || entry.en || detail;
}

// Locale used for Date formatting on the patient page.
function dateLocale() {
    return getLang() === 'ru' ? 'ru-RU' : 'en-GB';
}

// Switch language and re-render everything via a full reload.
function switchLang(lang) {
    if (lang === getLang()) return;
    setLang(lang);
    location.reload();
}

// Markup for the RU | EN toggle. Embedded directly in JS-rendered top bars,
// and injected into any static `#langSwitcher` placeholder on load.
function langSwitcherHTML() {
    const lang = getLang();
    return `
        <div class="lang-switcher">
            <button class="lang-btn ${lang === 'ru' ? 'active' : ''}" onclick="switchLang('ru')">RU</button>
            <span class="lang-sep">|</span>
            <button class="lang-btn ${lang === 'en' ? 'active' : ''}" onclick="switchLang('en')">EN</button>
        </div>`;
}

// Translate static HTML: `data-i18n` -> innerHTML, `data-i18n-ph` -> placeholder.
function applyStaticTranslations(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((el) => {
        el.innerHTML = t(el.getAttribute('data-i18n'));
    });
    root.querySelectorAll('[data-i18n-ph]').forEach((el) => {
        el.placeholder = t(el.getAttribute('data-i18n-ph'));
    });
}

document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.lang = getLang();
    const placeholder = document.getElementById('langSwitcher');
    if (placeholder) placeholder.innerHTML = langSwitcherHTML();
    applyStaticTranslations();
});