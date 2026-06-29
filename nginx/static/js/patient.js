// ========== CONSTANTS ==========
// Diagnosis enum values, used to build the diagnosis <select> dropdowns.
const DIAGNOSIS_KEYS = ['unknown', 'no_dr', 'mild_dr', 'moderate_dr', 'severe_dr', 'proliferative_dr'];
const REVIEW_DIAGNOSIS_KEYS = ['no_dr', 'mild_dr', 'moderate_dr', 'severe_dr', 'proliferative_dr'];

// ========== LABELS (i18n) ==========
const diagnosisLabel = (d) => t('diag_' + d);
const genderLabel = (g) => t('gender_' + g) || g;
const statusLabel = (s) => t('status_' + s) || s;

const POLLING = { INTERVAL: 2000, MAX_ATTEMPTS: 60 };

// ========== STATE ==========
let currentUser = null;
let patient = null;
let pollingInterval = null;
let selectedFile = null;
const patientId = window.location.pathname.split('/').pop();

// ========== API ==========
const fetchOpts = { credentials: 'include' };
const api = {
    getCurrentUser: () => fetch('/api/users/current', fetchOpts),
    getPatient: (id) => fetch(`/api/patients/${id}`, fetchOpts),
    getPatientTasks: (id) => fetch(`/api/patients/${id}/tasks`, fetchOpts),
    runMLTask: (pid, imageFile) => {
        const fd = new FormData();
        fd.append('image', imageFile);
        fd.append('patient_id', pid);
        return fetch('/api/ml-tasks/run', { ...fetchOpts, method: 'POST', body: fd });
    },
    getTaskStatus: (taskId) => fetch(`/api/ml-tasks/${taskId}`, fetchOpts),
    reviewTask: (taskId, conclusion) => fetch(`/api/ml-tasks/${taskId}/review`, {
        ...fetchOpts,
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doctor_conclusion: conclusion }),
    }),
    updatePatient: (id, data) => fetch(`/api/patients/${id}`, {
        ...fetchOpts,
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    logout: () => fetch('/api/sign-out', { method: 'POST', ...fetchOpts }),
};

// ========== TEMPLATES ==========
const getAvatarLetter = (u) => u ? u[0].toUpperCase() : '?';

function diagnosisBadge(d) {
    const label = d ? diagnosisLabel(d) : '—';
    return `<span class="diag-badge ${diagnosisBadgeClass(d)}">${label}</span>`;
}

function statusBadge(s) {
    const label = statusLabel(s);
    const cls = (s === 'completed' || s === 'reviewed') ? 'badge-success'
        : s === 'failed' ? 'badge-danger' : 'badge-neutral';
    return `<span class="status-badge ${cls}">${label}</span>`;
}

function formatDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    const locale = dateLocale();
    return d.toLocaleDateString(locale, { day: '2-digit', month: '2-digit', year: 'numeric' })
        + ' ' + d.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });
}

const templates = {
    topBar: (userContent) => `
        <div class="top-bar">
            <div class="top-bar-left">
                <a href="/" class="back-btn">${t('back_patients')}</a>
                <div class="logo">
                    <span class="logo-icon">👁️</span>
                    <span class="logo-text">eye</span>
                    <span class="logo-dot">.ai</span>
                </div>
            </div>
            <div class="top-bar-right">${langSwitcherHTML()}${userContent}</div>
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

    patientHeader: (p) => `
        <div class="patient-header card">
            <div class="patient-header-info">
                <h1 class="patient-full-name">${p.last_name} ${p.first_name}${p.middle_name ? ' ' + p.middle_name : ''}</h1>
                <div class="patient-meta">
                    <span>${p.age} ${t('years_old')}</span>
                    <span class="meta-sep">·</span>
                    <span>${genderLabel(p.gender)}</span>
                    <span class="meta-sep">·</span>
                    <span>${t('diagnosis_prefix')} ${diagnosisBadge(p.current_diagnosis)}</span>
                </div>
            </div>
            <button onclick="openEditPatientModal()" class="btn btn-ghost btn-sm">${t('edit')}</button>
        </div>`,

    editPatientModal: (p) => `
        <div class="modal-backdrop" id="editPatientBackdrop" onclick="closeEditPatientModal()">
            <div class="modal" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3 class="modal-title">${t('modal_edit_title')}</h3>
                    <button onclick="closeEditPatientModal()" class="modal-close">✕</button>
                </div>
                <div id="editPatientMessage"></div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">${t('f_last_name')}</label>
                        <input type="text" id="editLastName" class="form-input" value="${p.last_name}">
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('f_first_name')}</label>
                        <input type="text" id="editFirstName" class="form-input" value="${p.first_name}">
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">${t('f_middle_name')} <span class="text-muted">${t('optional')}</span></label>
                    <input type="text" id="editMiddleName" class="form-input" value="${p.middle_name || ''}">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">${t('f_age')}</label>
                        <input type="number" id="editAge" class="form-input" min="0" max="150" value="${p.age}">
                    </div>
                    <div class="form-group">
                        <label class="form-label">${t('f_gender')}</label>
                        <select id="editGender" class="form-input">
                            <option value="male" ${p.gender === 'male' ? 'selected' : ''}>${t('gender_male')}</option>
                            <option value="female" ${p.gender === 'female' ? 'selected' : ''}>${t('gender_female')}</option>
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">${t('diagnosis')}</label>
                    <select id="editDiagnosis" class="form-input">
                        ${DIAGNOSIS_KEYS.map((val) =>
                            `<option value="${val}" ${p.current_diagnosis === val ? 'selected' : ''}>${diagnosisLabel(val)}</option>`
                        ).join('')}
                    </select>
                </div>
                <div class="modal-footer">
                    <button onclick="closeEditPatientModal()" class="btn btn-ghost">${t('cancel')}</button>
                    <button onclick="submitEditPatient()" id="editPatientBtn" class="btn btn-primary">${t('save')}</button>
                </div>
            </div>
        </div>`,

    uploadCard: () => `
        <div class="card-gradient upload-card">
            <div class="ml-header">
                <span class="ml-title">${t('new_scan')}</span>
                <div class="badge-group">
                    <span class="badge">${t('badge_formats')}</span>
                    <span class="badge">${t('badge_size')}</span>
                </div>
            </div>
            <div class="ml-form">
                <div class="ml-input-group">
                    <label class="file-drop-area" id="fileDropArea">
                        <input type="file" id="imageInput" accept=".jpg,.jpeg,.png,.tiff,.tif" style="display:none">
                        <div id="fileDropContent" class="file-drop-content">
                            <span class="file-drop-icon">🔬</span>
                            <span class="file-drop-text">${t('drop_here')}</span>
                            <span class="file-drop-hint">${t('drop_hint')}</span>
                        </div>
                    </label>
                </div>
                <div class="ml-actions">
                    <button onclick="handleRunTask()" id="runTaskBtn" class="ml-btn" disabled>${t('analyze')}</button>
                </div>
            </div>
            <div id="uploadStatus"></div>
        </div>`,

    taskRow: (task) => `
        <tr>
            <td class="task-date">${formatDate(task.created_at)}</td>
            <td class="task-filename">${task.image_filename || '—'}</td>
            <td>${task.prediction ? diagnosisBadge(task.prediction.diagnosis) : '—'}</td>
            <td>${task.doctor_conclusion ? diagnosisBadge(task.doctor_conclusion) : '<span class="no-conclusion">—</span>'}</td>
            <td>${statusBadge(task.task_status)}</td>
            <td class="task-actions">
                ${task.task_status === 'completed'
                    ? `<button onclick="openReviewModal('${task.id}')" class="btn btn-primary btn-sm">${t('review_btn')}</button>`
                    : task.task_status === 'reviewed'
                    ? `<button onclick="openReviewModal('${task.id}')" class="btn btn-ghost btn-sm">${t('edit_btn')}</button>`
                    : ''}
            </td>
        </tr>`,

    taskTable: (tasks) => `
        <div class="section-header">
            <h2 class="section-title">${t('screening_history')}</h2>
        </div>
        ${tasks.length === 0
            ? `<div class="empty-state"><div class="empty-icon">🔬</div><p>${t('no_screenings')}</p></div>`
            : `<div class="table-container">
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>${t('th_date')}</th>
                            <th>${t('th_file')}</th>
                            <th>${t('th_ai')}</th>
                            <th>${t('th_doctor')}</th>
                            <th>${t('th_status')}</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>${tasks.map(templates.taskRow).join('')}</tbody>
                </table>
               </div>`
        }`,

    reviewModal: (taskId) => `
        <div class="modal-backdrop" id="reviewModalBackdrop" onclick="closeReviewModal()">
            <div class="modal" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3 class="modal-title">${t('modal_review_title')}</h3>
                    <button onclick="closeReviewModal()" class="modal-close">✕</button>
                </div>
                <div id="reviewModalMessage"></div>
                <div class="form-group">
                    <label class="form-label">${t('diagnosis')}</label>
                    <select id="reviewConclusion" class="form-input">
                        ${REVIEW_DIAGNOSIS_KEYS.map((val) =>
                            `<option value="${val}">${diagnosisLabel(val)}</option>`
                        ).join('')}
                    </select>
                </div>
                <div class="modal-footer">
                    <button onclick="closeReviewModal()" class="btn btn-ghost">${t('cancel')}</button>
                    <button onclick="submitReview('${taskId}')" id="reviewSubmitBtn" class="btn btn-primary">${t('save')}</button>
                </div>
            </div>
        </div>`,
};

// ========== FILE INPUT ==========
function initFileInput() {
    const input = document.getElementById('imageInput');
    const dropArea = document.getElementById('fileDropArea');
    if (!input || !dropArea) return;

    input.addEventListener('change', () => { if (input.files[0]) selectFile(input.files[0]); });
    dropArea.addEventListener('click', (e) => { if (e.target !== input) input.click(); });
    dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.classList.add('drag-over'); });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
    });
}

function selectFile(file) {
    selectedFile = file;
    const btn = document.getElementById('runTaskBtn');
    if (btn) btn.disabled = false;

    const content = document.getElementById('fileDropContent');
    if (!content) return;

    const info = `
        <span class="file-drop-text">${file.name}</span>
        <span class="file-drop-hint">${(file.size / 1024 / 1024).toFixed(2)} MB · ${t('click_to_change')}</span>`;

    // Browsers can't render TIFF in <img>, so only preview formats they support.
    const previewable = /^image\/(jpeg|png|webp|gif|bmp)$/i.test(file.type);
    if (previewable) {
        content.innerHTML = `<span class="file-drop-icon">🖼️</span>${info}`;
        const reader = new FileReader();
        reader.onload = (e) => {
            if (selectedFile !== file) return; // another file picked while reading
            content.innerHTML = `<img class="file-preview" src="${e.target.result}" alt="${file.name}">${info}`;
        };
        reader.readAsDataURL(file);
    } else {
        content.innerHTML = `<span class="file-drop-icon">📎</span>${info}`;
    }
}

// ========== ML TASK ==========
async function handleRunTask() {
    if (!selectedFile) return;
    const runBtn = document.getElementById('runTaskBtn');
    const statusEl = document.getElementById('uploadStatus');

    runBtn.disabled = true;
    runBtn.innerText = '...';
    statusEl.innerHTML = `<div class="upload-status processing"><div class="spinner-sm"></div><span>${t('analyzing')}</span></div>`;

    try {
        const response = await api.runMLTask(patientId, selectedFile);
        const result = await response.json();
        if (response.ok) {
            startPolling(result.id, runBtn, statusEl);
        } else {
            runBtn.disabled = false;
            runBtn.innerText = t('analyze');
            statusEl.innerHTML = `<div class="upload-status error">⚠️ ${translateApiError(result.detail) || t('failed_start')}</div>`;
        }
    } catch {
        runBtn.disabled = false;
        runBtn.innerText = t('analyze');
        statusEl.innerHTML = `<div class="upload-status error">⚠️ ${t('network_error')}</div>`;
    }
}

// ========== POLLING ==========
function startPolling(taskId, runBtn, statusEl) {
    stopPolling();
    let attempts = 0;
    pollingInterval = setInterval(async () => {
        attempts++;
        try {
            const response = await api.getTaskStatus(taskId);
            if (response.ok) {
                const task = await response.json();
                if (task.task_status === 'completed' || task.task_status === 'reviewed') {
                    stopPolling();
                    statusEl.innerHTML = '';
                    runBtn.disabled = false;
                    runBtn.innerText = t('analyze');
                    await refreshTaskList();
                } else if (task.task_status === 'failed') {
                    stopPolling();
                    statusEl.innerHTML = `<div class="upload-status error">⚠️ ${t('analysis_failed')}</div>`;
                    runBtn.disabled = false;
                    runBtn.innerText = t('analyze');
                }
            }
            if (attempts >= POLLING.MAX_ATTEMPTS) {
                stopPolling();
                statusEl.innerHTML = `<div class="upload-status error">⏱️ ${t('timeout')}</div>`;
                runBtn.disabled = false;
                runBtn.innerText = t('analyze');
            }
        } catch { /* ignore during polling */ }
    }, POLLING.INTERVAL);
}

function stopPolling() {
    if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }
}

async function refreshTaskList() {
    const response = await api.getPatientTasks(patientId);
    if (!response.ok) return;
    const tasks = await response.json();
    const el = document.getElementById('taskHistory');
    if (el) el.innerHTML = templates.taskTable(tasks);
}

// ========== REVIEW ==========
function openReviewModal(taskId) {
    closeReviewModal();
    document.body.insertAdjacentHTML('beforeend', templates.reviewModal(taskId));
}

function closeReviewModal() {
    document.getElementById('reviewModalBackdrop')?.remove();
}

async function submitReview(taskId) {
    const conclusion = document.getElementById('reviewConclusion').value;
    const btn = document.getElementById('reviewSubmitBtn');
    btn.disabled = true;
    btn.innerText = t('saving');

    const response = await api.reviewTask(taskId, conclusion);
    if (response.ok) {
        closeReviewModal();
        await refreshTaskList();
    } else {
        btn.disabled = false;
        btn.innerText = t('save');
        const result = await response.json();
        const msgEl = document.getElementById('reviewModalMessage');
        if (msgEl) msgEl.innerHTML = `<p class="message-error mb-3">${translateApiError(result.detail) || t('failed_save')}</p>`;
    }
}

// ========== EDIT PATIENT ==========
function openEditPatientModal() {
    closeEditPatientModal();
    document.body.insertAdjacentHTML('beforeend', templates.editPatientModal(patient));
}

function closeEditPatientModal() {
    document.getElementById('editPatientBackdrop')?.remove();
}

async function submitEditPatient() {
    const lastName = document.getElementById('editLastName').value.trim();
    const firstName = document.getElementById('editFirstName').value.trim();
    const middleName = document.getElementById('editMiddleName').value.trim();
    const age = parseInt(document.getElementById('editAge').value);
    const gender = document.getElementById('editGender').value;
    const diagnosis = document.getElementById('editDiagnosis').value;

    if (!lastName || !firstName || !age || !gender) {
        const msgEl = document.getElementById('editPatientMessage');
        if (msgEl) msgEl.innerHTML = `<p class="message-error mb-3">${t('fill_required')}</p>`;
        return;
    }

    const btn = document.getElementById('editPatientBtn');
    btn.disabled = true;
    btn.innerText = t('saving');

    const body = { first_name: firstName, last_name: lastName, age, gender, current_diagnosis: diagnosis };
    if (middleName) body.middle_name = middleName;

    const response = await api.updatePatient(patientId, body);
    if (response.ok) {
        patient = await response.json();
        closeEditPatientModal();
        const el = document.getElementById('patientHeaderContainer');
        if (el) el.innerHTML = templates.patientHeader(patient);
    } else {
        btn.disabled = false;
        btn.innerText = t('save');
        const result = await response.json();
        const msg = Array.isArray(result.detail) ? result.detail[0]?.msg : translateApiError(result.detail);
        const msgEl = document.getElementById('editPatientMessage');
        if (msgEl) msgEl.innerHTML = `<p class="message-error mb-3">${msg || t('failed_save')}</p>`;
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
async function init() {
    const content = document.getElementById('dashboard-content');
    content.innerHTML = `<div class="loading-state"><div class="loading-spinner"></div><p>${t('loading')}</p></div>`;

    try {
        const userResponse = await api.getCurrentUser();
        if (!userResponse.ok) { window.location.href = '/sign-in'; return; }
        currentUser = await userResponse.json();

        const patientResponse = await api.getPatient(patientId);
        if (!patientResponse.ok) { window.location.href = '/'; return; }
        patient = await patientResponse.json();

        const tasksResponse = await api.getPatientTasks(patientId);
        const tasks = tasksResponse.ok ? await tasksResponse.json() : [];

        content.innerHTML = `
            ${templates.topBar(templates.userSection(currentUser))}
            <div id="patientHeaderContainer">${templates.patientHeader(patient)}</div>
            ${templates.uploadCard()}
            <div id="taskHistory">${templates.taskTable(tasks)}</div>`;

        initFileInput();
    } catch {
        content.innerHTML = `<div class="error-state"><p class="text-error">${t('something_wrong')}</p>
            <button onclick="init()" class="btn btn-primary btn-sm mt-3">${t('try_again')}</button></div>`;
    }
}

document.addEventListener('DOMContentLoaded', init);
