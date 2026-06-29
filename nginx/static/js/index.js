// ========== LABELS (i18n) ==========
const diagnosisLabel = (d) => t('diag_' + d) || d;
const genderLabel = (g) => t('gender_' + g) || g;

function formatDateShort(iso) {
    if (!iso) return '';
    return new Date(iso).toLocaleDateString(dateLocale(), { day: '2-digit', month: '2-digit', year: 'numeric' });
}

function licenseBadge(s) {
    const cls = s === 'active' ? 'badge-success' : s === 'expired' ? 'badge-danger' : 'badge-neutral';
    return `<span class="diag-badge ${cls}">${t('license_' + s) || s}</span>`;
}

const hasAdminRole = (u) => (u.roles || []).includes('company_admin');

// "Иванов И.И." — last name + initials, falls back to username.
function formatDoctorName(u) {
    if (!u.last_name && !u.first_name) return u.username;
    const initials = [u.first_name, u.middle_name]
        .filter(Boolean)
        .map((n) => n[0].toUpperCase() + '.')
        .join(' ');
    return `${u.last_name} ${initials}`.trim();
}

const TRASH_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>';

// ========== STATE ==========
let currentUser = null;
let licensesCache = [];

// ========== API ==========
const fetchOpts = { credentials: 'include' };
const api = {
    getCurrentUser: () => fetch('/api/users/current', fetchOpts),
    getCompany: () => fetch('/api/company-admin/companies/', fetchOpts),
    getDoctors: () => fetch('/api/company-admin/users/', fetchOpts),
    createDoctor: (data) => fetch('/api/company-admin/users/', {
        ...fetchOpts,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    updateDoctorRole: (id, role) => fetch(`/api/company-admin/users/${id}/role`, {
        ...fetchOpts,
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role }),
    }),
    deleteDoctor: (id) => fetch(`/api/company-admin/users/${id}`, { ...fetchOpts, method: 'DELETE' }),
    getLicenses: () => fetch('/api/admin/licenses/', fetchOpts),
    updateLicense: (id, data) => fetch(`/api/admin/licenses/${id}`, {
        ...fetchOpts,
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    getPatients: () => fetch('/api/patients/', fetchOpts),
    createPatient: (data) => fetch('/api/patients/', {
        ...fetchOpts,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    logout: () => fetch('/api/sign-out', { method: 'POST', ...fetchOpts }),
};

// ========== TEMPLATES ==========
const getAvatarLetter = (u) => u ? u[0].toUpperCase() : '?';

const templates = {
    logo: () => `
        <div class="logo">
            <span class="logo-icon">👁️</span>
            <span class="logo-text">eye</span>
            <span class="logo-dot">.ai</span>
        </div>`,

    topBar: (content) => `
        <div class="top-bar">
            <div class="top-bar-left">${templates.logo()}</div>
            <div class="top-bar-right">${langSwitcherHTML()}${content}</div>
        </div>`,

    userSection: (user) => `
        <div class="user-section">
            <div class="avatar">${getAvatarLetter(user.username)}</div>
            <div class="user-info">
                <span class="user-name">${user.username}</span>
                <span class="user-email">${user.email}</span>
            </div>
            <button onclick="handleLogout()" id="logoutBtn" class="btn btn-outline-danger btn-sm">${t('logout')}</button>
        </div>`,

    diagnosisBadge: (d) => {
        const label = diagnosisLabel(d);
        return `<span class="diag-badge ${diagnosisBadgeClass(d)}">${label}</span>`;
    },

    patientRow: (p) => `
        <tr class="patient-row" onclick="window.location.href='/patients/${p.id}'">
            <td class="patient-name">${p.last_name} ${p.first_name}${p.middle_name ? ' ' + p.middle_name : ''}</td>
            <td>${p.age}</td>
            <td>${genderLabel(p.gender)}</td>
            <td>${templates.diagnosisBadge(p.current_diagnosis)}</td>
        </tr>`,

    patientTable: (patients) => `
        <div class="patients-header">
            <h2 class="patients-title">${t('patients_title')}</h2>
            <button onclick="openAddPatientModal()" class="btn btn-primary btn-sm">${t('add_patient')}</button>
        </div>
        ${patients.length === 0
            ? `<div class="empty-state"><div class="empty-icon">🏥</div><p>${t('no_patients')}</p></div>`
            : `<div class="table-container">
                <table class="patients-table">
                    <thead><tr><th>${t('th_name')}</th><th>${t('th_age')}</th><th>${t('th_gender')}</th><th>${t('th_diagnosis')}</th></tr></thead>
                    <tbody>${patients.map(templates.patientRow).join('')}</tbody>
                </table>
               </div>`
        }`,

    companyCard: (c) => {
        const used = c.used_scans || 0;
        const unlimited = c.max_scans === null || c.max_scans === undefined;
        const max = unlimited ? '∞' : c.max_scans;
        const remaining = unlimited ? '∞' : (c.remaining_scans ?? Math.max(c.max_scans - used, 0));
        const pct = unlimited ? 0 : (c.max_scans > 0 ? Math.min(100, Math.round(used / c.max_scans * 100)) : 0);
        const depleted = !unlimited && remaining === 0;
        return `
        <div class="company-card card">
            <div class="company-card-head">
                <div class="company-card-info">
                    <h2 class="company-name">${c.name}</h2>
                    <div class="company-meta">
                        ${licenseBadge(c.license_status)}
                        ${c.expired_at ? `<span class="company-expiry">${t('license_until')} ${formatDateShort(c.expired_at)}</span>` : ''}
                    </div>
                </div>
                <div class="scan-counter">
                    <span class="scan-counter-value ${depleted ? 'scan-counter-depleted' : ''}">${used} / ${max}</span>
                    <span class="scan-counter-label">${t('scans_label')}</span>
                </div>
            </div>
            <div class="scan-bar"><div class="scan-bar-fill ${depleted ? 'scan-bar-depleted' : ''}" style="width:${pct}%"></div></div>
            <div class="scan-remaining">${t('scans_remaining')}: <strong>${remaining}</strong></div>
        </div>`;
    },

    doctorsSection: (doctors) => `
        <div class="patients-header">
            <h2 class="patients-title">${t('doctors_title')}</h2>
            <button onclick="openAddDoctorModal()" class="btn btn-primary btn-sm">${t('add_doctor')}</button>
        </div>
        <div class="table-container">
            <table class="patients-table">
                <thead><tr><th>${t('th_name')}</th><th>Email</th><th>${t('th_role')}</th><th></th></tr></thead>
                <tbody>${doctors.map(templates.doctorRow).join('')}</tbody>
            </table>
        </div>`,

    doctorRow: (u) => {
        const admin = hasAdminRole(u);
        const isSelf = currentUser && u.id === currentUser.id;
        const roleCls = admin ? 'badge-warning' : 'badge-neutral';
        return `
        <tr>
            <td class="patient-name">${formatDoctorName(u)}${isSelf ? ` <span class="doctor-you">${t('you')}</span>` : ''}</td>
            <td class="doctor-email">${u.email}</td>
            <td><span class="diag-badge ${roleCls}">${admin ? t('role_admin') : t('role_doctor')}</span></td>
            <td>
                <div class="doctor-actions">
                    ${isSelf ? '' : `
                        <button onclick="handleToggleRole('${u.id}', ${admin})" class="btn btn-ghost btn-sm">${admin ? t('demote') : t('promote')}</button>
                        <button onclick="handleDeleteDoctor('${u.id}')" class="icon-btn icon-btn-danger" title="${t('delete')}" aria-label="${t('delete')}">${TRASH_ICON}</button>`}
                </div>
            </td>
        </tr>`;
    },

    addDoctorModal: () => `
        <div class="modal-backdrop" id="doctorModalBackdrop" onclick="closeAddDoctorModal()">
            <div class="modal" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3 class="modal-title">${t('add_doctor')}</h3>
                    <button onclick="closeAddDoctorModal()" class="modal-close">✕</button>
                </div>
                <div id="doctorModalMessage"></div>
                <form id="addDoctorForm" onsubmit="return false;">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">${t('f_last_name')}</label>
                            <input type="text" id="doctorLastName" class="form-input" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">${t('f_first_name')}</label>
                            <input type="text" id="doctorFirstName" class="form-input" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('f_middle_name')} <span class="text-muted">${t('optional')}</span></label>
                        <input type="text" id="doctorMiddleName" class="form-input">
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('username')}</label>
                        <input type="text" id="doctorUsername" class="form-input" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Email</label>
                        <input type="email" id="doctorEmail" class="form-input" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('password')}</label>
                        <input type="password" id="doctorPassword" class="form-input" required>
                    </div>
                    <div class="modal-footer">
                        <button type="button" onclick="closeAddDoctorModal()" class="btn btn-ghost">${t('cancel')}</button>
                        <button type="submit" id="addDoctorBtn" onclick="handleAddDoctor()" class="btn btn-primary">${t('add_doctor_btn')}</button>
                    </div>
                </form>
            </div>
        </div>`,

    licensesSection: (licenses) => `
        <div class="patients-header">
            <h2 class="patients-title">${t('licenses_title')}</h2>
        </div>
        <div class="table-container">
            <table class="patients-table">
                <thead><tr><th>${t('th_company')}</th><th>${t('th_status')}</th><th>${t('scans_label')}</th><th>${t('th_expiry')}</th><th></th></tr></thead>
                <tbody>${licenses.map(templates.licenseRow).join('')}</tbody>
            </table>
        </div>`,

    licenseRow: (lic) => `
        <tr>
            <td class="patient-name">${lic.company_name}</td>
            <td>${licenseBadge(lic.license_status ?? lic.status)}</td>
            <td>${lic.used_scans} / ${lic.max_scans ?? '∞'}</td>
            <td>${lic.expired_at ? formatDateShort(lic.expired_at) : '—'}</td>
            <td>
                <div class="doctor-actions">
                    <button onclick="openEditLicenseModal('${lic.id}')" class="btn btn-ghost btn-sm">${t('edit')}</button>
                </div>
            </td>
        </tr>`,

    editLicenseModal: (lic) => `
        <div class="modal-backdrop" id="licenseModalBackdrop" onclick="closeEditLicenseModal()">
            <div class="modal" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3 class="modal-title">${lic.company_name}</h3>
                    <button onclick="closeEditLicenseModal()" class="modal-close">✕</button>
                </div>
                <div id="licenseModalMessage"></div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">${t('th_status')}</label>
                        <select id="licenseStatus" class="form-input">
                            ${['demo', 'active', 'expired'].map((s) =>
                                `<option value="${s}" ${lic.status === s ? 'selected' : ''}>${t('license_' + s)}</option>`
                            ).join('')}
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('max_scans_label')}</label>
                        <input type="number" id="licenseMaxScans" class="form-input" min="1" placeholder="∞" value="${lic.max_scans ?? ''}">
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">${t('th_expiry')} <span class="text-muted">${t('optional')}</span></label>
                    <input type="date" id="licenseExpiredAt" class="form-input" value="${lic.expired_at ? lic.expired_at.slice(0, 10) : ''}">
                </div>
                <div class="form-group">
                    <label class="form-label">${t('notes_label')} <span class="text-muted">${t('optional')}</span></label>
                    <textarea id="licenseNotes" class="form-input" rows="2">${lic.notes || ''}</textarea>
                </div>
                <div class="modal-footer">
                    <button onclick="closeEditLicenseModal()" class="btn btn-ghost">${t('cancel')}</button>
                    <button onclick="submitEditLicense('${lic.id}')" id="licenseSubmitBtn" class="btn btn-primary">${t('save')}</button>
                </div>
            </div>
        </div>`,

    addPatientModal: () => `
        <div class="modal-backdrop" id="modalBackdrop" onclick="closeAddPatientModal()">
            <div class="modal" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3 class="modal-title">${t('modal_add_title')}</h3>
                    <button onclick="closeAddPatientModal()" class="modal-close">✕</button>
                </div>
                <div id="modalMessage"></div>
                <form id="addPatientForm" onsubmit="return false;">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">${t('f_last_name')}</label>
                            <input type="text" id="patientLastName" class="form-input" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">${t('f_first_name')}</label>
                            <input type="text" id="patientFirstName" class="form-input" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('f_middle_name')} <span class="text-muted">${t('optional')}</span></label>
                        <input type="text" id="patientMiddleName" class="form-input">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">${t('f_age')}</label>
                            <input type="number" id="patientAge" class="form-input" min="0" max="150" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">${t('f_gender')}</label>
                            <select id="patientGender" class="form-input" required>
                                <option value="">${t('select_placeholder')}</option>
                                <option value="male">${t('gender_male')}</option>
                                <option value="female">${t('gender_female')}</option>
                            </select>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" onclick="closeAddPatientModal()" class="btn btn-ghost">${t('cancel')}</button>
                        <button type="submit" id="addPatientBtn" onclick="handleAddPatient()" class="btn btn-primary">${t('btn_add_patient')}</button>
                    </div>
                </form>
            </div>
        </div>`,
};

// ========== COMPANY CARD ==========
async function renderCompanyCard() {
    const el = document.getElementById('companyCard');
    if (!el) return;
    try {
        const response = await api.getCompany();
        if (!response.ok) return; // not an admin / no license — skip silently
        const data = await response.json();
        el.innerHTML = templates.companyCard(data);
    } catch { /* card is non-critical, skip on error */ }
}

// ========== DOCTORS ==========
async function renderDoctorsSection() {
    const el = document.getElementById('doctorsSection');
    if (!el) return;
    try {
        const response = await api.getDoctors();
        if (!response.ok) return;
        const doctors = await response.json();
        el.innerHTML = templates.doctorsSection(doctors);
    } catch { /* non-critical section */ }
}

function openAddDoctorModal() {
    if (!document.getElementById('doctorModalBackdrop')) {
        document.body.insertAdjacentHTML('beforeend', templates.addDoctorModal());
    }
}

function closeAddDoctorModal() {
    document.getElementById('doctorModalBackdrop')?.remove();
}

function showDoctorMessage(text, type) {
    const el = document.getElementById('doctorModalMessage');
    if (el) el.innerHTML = `<p class="message-${type} mb-3">${text}</p>`;
}

async function handleAddDoctor() {
    const lastName = document.getElementById('doctorLastName').value.trim();
    const firstName = document.getElementById('doctorFirstName').value.trim();
    const middleName = document.getElementById('doctorMiddleName').value.trim();
    const username = document.getElementById('doctorUsername').value.trim();
    const email = document.getElementById('doctorEmail').value.trim();
    const password = document.getElementById('doctorPassword').value;

    if (!lastName || !firstName || !username || !email || !password) {
        showDoctorMessage(t('fill_required'), 'error');
        return;
    }

    const btn = document.getElementById('addDoctorBtn');
    btn.disabled = true;
    btn.innerText = t('adding');

    const body = { last_name: lastName, first_name: firstName, username, email, password };
    if (middleName) body.middle_name = middleName;

    const response = await api.createDoctor(body);
    const result = await response.json();

    if (response.ok) {
        closeAddDoctorModal();
        await renderDoctorsSection();
    } else {
        btn.disabled = false;
        btn.innerText = t('add_doctor_btn');
        const msg = Array.isArray(result.detail) ? result.detail[0]?.msg : translateApiError(result.detail);
        showDoctorMessage(msg || t('failed_save'), 'error');
    }
}

async function handleToggleRole(id, isAdmin) {
    const response = await api.updateDoctorRole(id, isAdmin ? 'user' : 'company_admin');
    if (response.ok) await renderDoctorsSection();
}

async function handleDeleteDoctor(id) {
    if (!confirm(t('confirm_delete_doctor'))) return;
    const response = await api.deleteDoctor(id);
    if (response.ok || response.status === 204) await renderDoctorsSection();
}

// ========== LICENSES (super admin) ==========
async function renderLicensesSection() {
    const el = document.getElementById('licensesSection');
    if (!el) return;
    try {
        const response = await api.getLicenses();
        if (!response.ok) return;
        licensesCache = await response.json();
        el.innerHTML = templates.licensesSection(licensesCache);
    } catch { /* non-critical section */ }
}

function openEditLicenseModal(id) {
    const lic = licensesCache.find((l) => l.id === id);
    if (!lic) return;
    closeEditLicenseModal();
    document.body.insertAdjacentHTML('beforeend', templates.editLicenseModal(lic));
}

function closeEditLicenseModal() {
    document.getElementById('licenseModalBackdrop')?.remove();
}

async function submitEditLicense(id) {
    const btn = document.getElementById('licenseSubmitBtn');
    btn.disabled = true;
    btn.innerText = t('saving');

    const maxScansRaw = document.getElementById('licenseMaxScans').value.trim();
    const expiredAt = document.getElementById('licenseExpiredAt').value;
    const notes = document.getElementById('licenseNotes').value.trim();

    const body = {
        status: document.getElementById('licenseStatus').value,
        notes: notes,
    };
    // empty fields are left unchanged; a number sets the scan cap
    if (maxScansRaw) body.max_scans = parseInt(maxScansRaw);
    if (expiredAt) body.expired_at = `${expiredAt}T00:00:00`;

    const response = await api.updateLicense(id, body);
    if (response.ok) {
        closeEditLicenseModal();
        await renderLicensesSection();
    } else {
        btn.disabled = false;
        btn.innerText = t('save');
        const result = await response.json();
        const msg = Array.isArray(result.detail) ? result.detail[0]?.msg : translateApiError(result.detail);
        const msgEl = document.getElementById('licenseModalMessage');
        if (msgEl) msgEl.innerHTML = `<p class="message-error mb-3">${msg || t('failed_save')}</p>`;
    }
}

// ========== PATIENT LIST ==========
async function renderPatientList() {
    const response = await api.getPatients();
    const patients = response.ok ? await response.json() : [];
    const el = document.getElementById('patientList');
    if (el) el.innerHTML = templates.patientTable(patients);
}

// ========== ADD PATIENT MODAL ==========
function openAddPatientModal() {
    if (!document.getElementById('modalBackdrop')) {
        document.body.insertAdjacentHTML('beforeend', templates.addPatientModal());
    }
}

function closeAddPatientModal() {
    document.getElementById('modalBackdrop')?.remove();
}

async function handleAddPatient() {
    const lastName = document.getElementById('patientLastName').value.trim();
    const firstName = document.getElementById('patientFirstName').value.trim();
    const middleName = document.getElementById('patientMiddleName').value.trim();
    const age = parseInt(document.getElementById('patientAge').value);
    const gender = document.getElementById('patientGender').value;

    if (!lastName || !firstName || !age || !gender) {
        showModalMessage(t('fill_required'), 'error');
        return;
    }

    const btn = document.getElementById('addPatientBtn');
    btn.disabled = true;
    btn.innerText = t('adding');

    const body = { first_name: firstName, last_name: lastName, age, gender };
    if (middleName) body.middle_name = middleName;

    const response = await api.createPatient(body);
    const result = await response.json();

    if (response.ok) {
        closeAddPatientModal();
        window.location.href = `/patients/${result.id}`;
    } else {
        btn.disabled = false;
        btn.innerText = t('btn_add_patient');
        const msg = Array.isArray(result.detail) ? result.detail[0]?.msg : translateApiError(result.detail);
        showModalMessage(msg || t('failed_add_patient'), 'error');
    }
}

function showModalMessage(text, type) {
    const el = document.getElementById('modalMessage');
    if (el) el.innerHTML = `<p class="message-${type} mb-3">${text}</p>`;
}

// ========== USER PROFILE ==========
async function loadUserProfile() {
    const content = document.getElementById('dashboard-content');
    content.innerHTML = `<div class="loading-state"><div class="loading-spinner"></div><p>${t('loading')}</p></div>`;
    try {
        const response = await api.getCurrentUser();
        if (response.ok) {
            currentUser = await response.json();
            const roles = currentUser.roles || [];
            const isSuperAdmin = roles.includes('super_admin');
            const isCompanyAdmin = roles.includes('company_admin');

            if (isSuperAdmin) {
                // super admin dashboard is the license management panel only
                content.innerHTML = templates.topBar(templates.userSection(currentUser))
                    + '<div id="licensesSection"></div>';
                await renderLicensesSection();
            } else {
                content.innerHTML = templates.topBar(templates.userSection(currentUser))
                    + (isCompanyAdmin ? '<div id="companyCard"></div><div id="doctorsSection"></div>' : '')
                    + '<div id="patientList"></div>';
                if (isCompanyAdmin) {
                    await renderCompanyCard();
                    await renderDoctorsSection();
                }
                await renderPatientList();
            }
        } else if (response.status === 401) {
            window.location.href = '/sign-in';
        } else {
            throw new Error();
        }
    } catch {
        content.innerHTML = `<div class="error-state"><p class="text-error">${t('something_wrong')}</p>
            <button onclick="loadUserProfile()" class="btn btn-primary btn-sm mt-3">${t('try_again')}</button></div>`;
    }
}

// ========== LOGOUT ==========
async function handleLogout() {
    const btn = document.getElementById('logoutBtn');
    const original = btn.innerText;
    try {
        btn.disabled = true;
        btn.innerText = '...';
        const response = await api.logout();
        if (response.ok) setTimeout(() => window.location.href = '/', 500);
    } catch {
        btn.disabled = false;
        btn.innerText = original;
    }
}

// ========== INIT ==========
document.addEventListener('DOMContentLoaded', loadUserProfile);
