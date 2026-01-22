// ==================== Utility Functions ====================

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

// ==================== API Key Functions ====================

async function fetchStatus() {
  try {
    const url = new URL("/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 2000);
    if (!resp.ok) throw new Error("status error");
    return await resp.json();
  } catch (e) {
    return { has_key: false, error: true };
  }
}

async function waitForStatus(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function validateKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/validate_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "validation_failed");
  }
  return data;
}

async function saveKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/openai_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

// ==================== Security API Functions ====================

async function getSecurityStatus() {
  try {
    const url = new URL("/security/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 2000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function armSecurity() {
  const resp = await fetch("/security/arm", { method: "POST" });
  if (!resp.ok) throw new Error("Failed to arm");
  return await resp.json();
}

async function disarmSecurity() {
  const resp = await fetch("/security/disarm", { method: "POST" });
  if (!resp.ok) throw new Error("Failed to disarm");
  return await resp.json();
}

async function saveSecuritySettings(settings) {
  const resp = await fetch("/security/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!resp.ok) throw new Error("Failed to save settings");
  return await resp.json();
}

async function getFaces() {
  try {
    const url = new URL("/security/faces", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 2000);
    if (!resp.ok) return { faces: [] };
    return await resp.json();
  } catch (e) {
    return { faces: [] };
  }
}

async function enrollFace(name, files) {
  const formData = new FormData();
  formData.append("name", name);
  for (const file of files) {
    formData.append("images", file);
  }
  const resp = await fetch("/security/faces", {
    method: "POST",
    body: formData,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || "Enrollment failed");
  }
  return data;
}

async function deleteFace(name) {
  const resp = await fetch(`/security/faces/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || "Delete failed");
  }
  return await resp.json();
}

// ==================== Security UI Functions ====================

function updateSecurityUI(status) {
  const statusChip = document.getElementById("security-status-chip");
  const armBtn = document.getElementById("arm-btn");
  const disarmBtn = document.getElementById("disarm-btn");

  if (!status) {
    statusChip.textContent = "Offline";
    statusChip.className = "chip";
    return;
  }

  if (status.armed) {
    statusChip.textContent = "🔴 ARMED";
    statusChip.className = "chip chip-armed";
    armBtn.classList.add("active");
    disarmBtn.classList.remove("active");
  } else {
    statusChip.textContent = "🟢 DISARMED";
    statusChip.className = "chip chip-disarmed";
    armBtn.classList.remove("active");
    disarmBtn.classList.add("active");
  }
}

function updateSettingsUI(settings) {
  if (!settings) return;

  const recognitionHz = document.getElementById("recognition-hz");
  const confidenceThreshold = document.getElementById("confidence-threshold");
  const unknownCooldown = document.getElementById("unknown-cooldown");
  const greetingCooldown = document.getElementById("greeting-cooldown");

  recognitionHz.value = settings.recognition_hz || 2;
  confidenceThreshold.value = (settings.confidence_threshold || 0.6) * 100;
  unknownCooldown.value = settings.unknown_cooldown_min || 5;
  greetingCooldown.value = settings.greeting_cooldown_min || 10;

  // Update display values
  document.getElementById("recognition-hz-value").textContent = `${recognitionHz.value} Hz`;
  document.getElementById("confidence-threshold-value").textContent = `${confidenceThreshold.value}%`;
  document.getElementById("unknown-cooldown-value").textContent = `${unknownCooldown.value} min`;
  document.getElementById("greeting-cooldown-value").textContent = `${greetingCooldown.value} min`;
}

function renderFacesList(faces) {
  const container = document.getElementById("faces-list");

  if (!faces || faces.length === 0) {
    container.innerHTML = '<div class="faces-empty"><p class="muted">No faces enrolled yet.</p></div>';
    return;
  }

  container.innerHTML = faces.map(face => `
    <div class="face-item" data-name="${face.name}">
      <div class="face-info">
        <div class="face-avatar">${face.name.charAt(0).toUpperCase()}</div>
        <div>
          <div class="face-name">${face.name}</div>
          <div class="face-meta">${face.image_count} image${face.image_count !== 1 ? 's' : ''}</div>
        </div>
      </div>
      <button class="face-delete ghost" onclick="handleDeleteFace('${face.name}')">Delete</button>
    </div>
  `).join("");
}

async function handleDeleteFace(name) {
  if (!confirm(`Delete face "${name}"?`)) return;

  const statusEl = document.getElementById("security-status");
  statusEl.textContent = "Deleting...";
  statusEl.className = "status";

  try {
    await deleteFace(name);
    statusEl.textContent = `Deleted ${name}`;
    statusEl.className = "status ok";
    // Refresh faces list
    const data = await getFaces();
    renderFacesList(data.faces);
  } catch (e) {
    statusEl.textContent = `Failed to delete: ${e.message}`;
    statusEl.className = "status error";
  }
}

// Make handleDeleteFace available globally for onclick
window.handleDeleteFace = handleDeleteFace;

// ==================== Main Initialization ====================

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);

  // API Key elements
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");

  // Security elements
  const armBtn = document.getElementById("arm-btn");
  const disarmBtn = document.getElementById("disarm-btn");
  const armStatus = document.getElementById("arm-status");
  const saveSettingsBtn = document.getElementById("save-settings-btn");
  const enrollBtn = document.getElementById("enroll-btn");
  const enrollName = document.getElementById("enroll-name");
  const enrollImages = document.getElementById("enroll-images");
  const enrollStatus = document.getElementById("enroll-status");
  const fileCount = document.getElementById("file-count");
  const securityStatus = document.getElementById("security-status");

  // Slider elements
  const recognitionHz = document.getElementById("recognition-hz");
  const confidenceThreshold = document.getElementById("confidence-threshold");
  const unknownCooldown = document.getElementById("unknown-cooldown");
  const greetingCooldown = document.getElementById("greeting-cooldown");

  // ==================== Security Setup ====================

  // Initialize security UI
  const secStatus = await getSecurityStatus();
  updateSecurityUI(secStatus);
  updateSettingsUI(secStatus);

  // Load faces
  const facesData = await getFaces();
  renderFacesList(facesData.faces);

  // Arm/Disarm handlers
  armBtn.addEventListener("click", async () => {
    armStatus.textContent = "Arming...";
    armStatus.className = "status";
    try {
      const result = await armSecurity();
      updateSecurityUI({ armed: true });
      armStatus.textContent = result.status || "Armed";
      armStatus.className = "status ok";
    } catch (e) {
      armStatus.textContent = "Failed to arm";
      armStatus.className = "status error";
    }
  });

  disarmBtn.addEventListener("click", async () => {
    armStatus.textContent = "Disarming...";
    armStatus.className = "status";
    try {
      const result = await disarmSecurity();
      updateSecurityUI({ armed: false });
      armStatus.textContent = result.status || "Disarmed";
      armStatus.className = "status ok";
    } catch (e) {
      armStatus.textContent = "Failed to disarm";
      armStatus.className = "status error";
    }
  });

  // Slider change handlers
  recognitionHz.addEventListener("input", () => {
    document.getElementById("recognition-hz-value").textContent = `${recognitionHz.value} Hz`;
  });
  confidenceThreshold.addEventListener("input", () => {
    document.getElementById("confidence-threshold-value").textContent = `${confidenceThreshold.value}%`;
  });
  unknownCooldown.addEventListener("input", () => {
    document.getElementById("unknown-cooldown-value").textContent = `${unknownCooldown.value} min`;
  });
  greetingCooldown.addEventListener("input", () => {
    document.getElementById("greeting-cooldown-value").textContent = `${greetingCooldown.value} min`;
  });

  // Save settings handler
  saveSettingsBtn.addEventListener("click", async () => {
    securityStatus.textContent = "Saving settings...";
    securityStatus.className = "status";
    try {
      await saveSecuritySettings({
        recognition_hz: parseFloat(recognitionHz.value),
        confidence_threshold: parseFloat(confidenceThreshold.value) / 100,
        unknown_cooldown_min: parseFloat(unknownCooldown.value),
        greeting_cooldown_min: parseFloat(greetingCooldown.value),
      });
      securityStatus.textContent = "Settings saved";
      securityStatus.className = "status ok";
    } catch (e) {
      securityStatus.textContent = "Failed to save settings";
      securityStatus.className = "status error";
    }
  });

  // File input handler
  enrollImages.addEventListener("change", () => {
    const count = enrollImages.files.length;
    fileCount.textContent = count === 0 ? "No files selected" : `${count} file${count !== 1 ? 's' : ''} selected`;
  });

  // Enroll handler
  enrollBtn.addEventListener("click", async () => {
    const name = enrollName.value.trim();
    const files = enrollImages.files;

    if (!name) {
      enrollStatus.textContent = "Please enter a name";
      enrollStatus.className = "status warn";
      return;
    }
    if (!files || files.length === 0) {
      enrollStatus.textContent = "Please select at least one image";
      enrollStatus.className = "status warn";
      return;
    }

    enrollStatus.textContent = "Enrolling...";
    enrollStatus.className = "status";

    try {
      const result = await enrollFace(name, files);
      enrollStatus.textContent = result.message || "Enrolled successfully";
      enrollStatus.className = "status ok";
      // Clear form
      enrollName.value = "";
      enrollImages.value = "";
      fileCount.textContent = "No files selected";
      // Refresh faces list
      const data = await getFaces();
      renderFacesList(data.faces);
    } catch (e) {
      enrollStatus.textContent = e.message || "Enrollment failed";
      enrollStatus.className = "status error";
    }
  });

  // ==================== API Key Setup ====================

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);

  const st = (await waitForStatus()) || { has_key: false };
  if (st.has_key) {
    statusEl.textContent = "";
    show(configuredPanel, true);
  }

  changeKeyBtn.addEventListener("click", () => {
    show(configuredPanel, false);
    show(formPanel, true);
    input.value = "";
    statusEl.textContent = "";
    statusEl.className = "status";
  });

  input.addEventListener("input", () => {
    input.classList.remove("error");
  });

  saveBtn.addEventListener("click", async () => {
    const key = input.value.trim();
    if (!key) {
      statusEl.textContent = "Please enter a valid key.";
      statusEl.className = "status warn";
      input.classList.add("error");
      return;
    }
    statusEl.textContent = "Validating API key...";
    statusEl.className = "status";
    input.classList.remove("error");
    try {
      const validation = await validateKey(key);
      if (!validation.valid) {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
        statusEl.className = "status error";
        input.classList.add("error");
        return;
      }
      statusEl.textContent = "Key valid! Saving...";
      statusEl.className = "status ok";
      await saveKey(key);
      statusEl.textContent = "Saved. Reloading…";
      statusEl.className = "status ok";
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (e.message === "invalid_api_key") {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
      } else {
        statusEl.textContent = "Failed to validate/save key. Please try again.";
      }
      statusEl.className = "status error";
    }
  });

  if (!st.has_key) {
    statusEl.textContent = "";
    show(formPanel, true);
  }

  show(loading, false);
}

window.addEventListener("DOMContentLoaded", init);
