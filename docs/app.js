(function () {
  const model = window.PCS_MODEL;
  const form = document.getElementById("predictorForm");
  const riskValue = document.getElementById("riskValue");
  const riskBar = document.getElementById("riskBar");
  const riskBand = document.getElementById("riskBand");
  const sampleSelect = document.getElementById("sampleSelect");
  const currentSample = document.getElementById("currentSample");
  const languageButtons = document.querySelectorAll(".lang-button");

  let language = "zh";
  let lastProbability = null;

  const uiText = {
    zh: {
      pageTitle: "PCS 风险预测",
      title: "PCS 风险预测",
      subtitle: "基于本项目随机森林模型的浏览器本地计算版本。",
      riskLabel: "预测概率",
      initialRisk: "请填写变量后计算",
      lowRisk: "较低风险：低于当前模型阳性阈值",
      midRisk: "中等风险：建议结合临床信息复核",
      highRisk: "高风险：达到当前模型阳性阈值",
      modelDetails: "模型信息",
      model: "模型",
      auc: "内部验证 AUC",
      sampleSize: "样本量",
      notice: "仅用于科研展示和内部验证，不能替代临床诊断或治疗决策。正式应用前仍需外部验证、校准和伦理/隐私审查。",
      sampleCaption: "内置样本",
      samplePlaceholder: "选择内置样本",
      customInput: "自定义输入",
      reset: "重置",
      predict: "计算风险",
      rangeHint: "参考范围",
      selectHint: "请选择最接近的分类",
    },
    en: {
      pageTitle: "PCS Risk Prediction",
      title: "PCS Risk Prediction",
      subtitle: "Browser-based local risk calculation using the project Random Forest model.",
      riskLabel: "Predicted risk",
      initialRisk: "Enter variables to calculate risk",
      lowRisk: "Lower risk: below the current positive threshold",
      midRisk: "Intermediate risk: review with clinical context",
      highRisk: "High risk: reaches the current positive threshold",
      modelDetails: "Model details",
      model: "Model",
      auc: "Internal validation AUC",
      sampleSize: "Sample size",
      notice: "For research demonstration and internal validation only. This tool does not replace clinical diagnosis or treatment decisions. External validation, calibration, ethics review, and privacy review are required before formal use.",
      sampleCaption: "Built-in profile",
      samplePlaceholder: "Select a built-in profile",
      customInput: "Custom input",
      reset: "Reset",
      predict: "Calculate risk",
      rangeHint: "Reference range",
      selectHint: "Select the closest category",
    },
  };

  const valueLabels = {
    zh: {
      "0": "否",
      "1": "是",
      male: "男",
      female: "女",
      mental_labor: "脑力劳动",
      manual_labor: "体力劳动",
      unemployed: "无业/未就业",
      junior_high_or_below: "初中及以下",
      high_school: "高中/中专",
      college_or_above: "大专及以上",
      none: "无",
      very_short: "很短",
      short: "短",
      medium: "中等",
      long: "长",
      daily: "每日",
      weekly: "每周",
      monthly: "每月",
      occasional: "偶发",
      single: "单发",
      multiple: "多发",
      no_stone: "未见结石",
      gallbladder_body: "胆囊体部",
      gallbladder_fundus: "胆囊底部",
      gallbladder_neck: "胆囊颈部",
      cystic_duct: "胆囊管",
      left_intrahepatic_bile_duct: "左肝内胆管",
      distal_common_bile_duct: "胆总管远端",
    },
    en: {
      "0": "No",
      "1": "Yes",
      male: "Male",
      female: "Female",
      mental_labor: "Mental labor",
      manual_labor: "Manual labor",
      unemployed: "Unemployed",
      junior_high_or_below: "Junior high or below",
      high_school: "High school",
      college_or_above: "College or above",
      none: "None",
      very_short: "Very short",
      short: "Short",
      medium: "Medium",
      long: "Long",
      daily: "Daily",
      weekly: "Weekly",
      monthly: "Monthly",
      occasional: "Occasional",
      single: "Single",
      multiple: "Multiple",
      no_stone: "No stone",
      gallbladder_body: "Gallbladder body",
      gallbladder_fundus: "Gallbladder fundus",
      gallbladder_neck: "Gallbladder neck",
      cystic_duct: "Cystic duct",
      left_intrahepatic_bile_duct: "Left intrahepatic bile duct",
      distal_common_bile_duct: "Distal common bile duct",
    },
  };

  const sampleProfiles = [
    { id: "low", label: { zh: "低风险样本", en: "Lower-risk profile" }, values: { age: 45, sex: "female", occupation: "mental_labor", height_cm: 164, weight_kg: 62, bmi: 23, education_level: "college_or_above", smoking: 0, alcohol_use: 0, symptom_duration: "none", abdominal_pain: 0, pain_frequency: "none", radiating_pain: 0, meal_related_pain: 0, hypertension: 0, hyperlipidemia: 0, diabetes: 0, anxiety_depression: 0, prior_abdominal_surgery: 0, gallbladder_wall_thickening: 0, max_stone_diameter_mm: 7, stone_number: "single", stone_location: "gallbladder_body", common_bile_duct_diameter_mm: 5, fatty_liver: 0, gallbladder_atrophy: 0, alt: 18, ast: 19, alp: 72, ggt: 20, total_bilirubin: 10, total_bile_acid: 2, total_cholesterol: 4.4, triglyceride: 1.1, alpha_fetoprotein: 2.3, ca199: 6 } },
    { id: "baseline", label: { zh: "典型中位样本", en: "Median profile" }, values: {} },
    { id: "symptom", label: { zh: "症状偏重样本", en: "Symptom-heavy profile" }, values: { age: 58, sex: "male", occupation: "manual_labor", height_cm: 168, weight_kg: 67, bmi: 24, education_level: "junior_high_or_below", smoking: 1, alcohol_use: 0, symptom_duration: "very_short", abdominal_pain: 1, pain_frequency: "daily", radiating_pain: 1, meal_related_pain: 1, hypertension: 0, hyperlipidemia: 0, diabetes: 0, anxiety_depression: 0, prior_abdominal_surgery: 0, gallbladder_wall_thickening: 1, max_stone_diameter_mm: 12, stone_number: "multiple", stone_location: "gallbladder_body", common_bile_duct_diameter_mm: 6, fatty_liver: 0, gallbladder_atrophy: 0, alt: 38, ast: 32, alp: 105, ggt: 58, total_bilirubin: 16, total_bile_acid: 8, total_cholesterol: 4.8, triglyceride: 1.5, alpha_fetoprotein: 2.8, ca199: 14 } },
    { id: "lab", label: { zh: "实验室异常样本", en: "Laboratory-abnormal profile" }, values: { age: 63, sex: "female", occupation: "unemployed", height_cm: 158, weight_kg: 60, bmi: 24, education_level: "junior_high_or_below", smoking: 0, alcohol_use: 0, symptom_duration: "short", abdominal_pain: 1, pain_frequency: "weekly", radiating_pain: 0, meal_related_pain: 0, hypertension: 1, hyperlipidemia: 1, diabetes: 0, anxiety_depression: 0, prior_abdominal_surgery: 0, gallbladder_wall_thickening: 1, max_stone_diameter_mm: 14, stone_number: "multiple", stone_location: "gallbladder_fundus", common_bile_duct_diameter_mm: 8, fatty_liver: 1, gallbladder_atrophy: 0, alt: 170, ast: 115, alp: 190, ggt: 320, total_bilirubin: 38, total_bile_acid: 45, total_cholesterol: 5.6, triglyceride: 2.1, alpha_fetoprotein: 3.5, ca199: 85 } },
    { id: "high", label: { zh: "高风险样本", en: "High-risk profile" }, values: { age: 70, sex: "male", occupation: "unemployed", height_cm: 166, weight_kg: 58, bmi: 21, education_level: "junior_high_or_below", smoking: 1, alcohol_use: 1, symptom_duration: "very_short", abdominal_pain: 1, pain_frequency: "daily", radiating_pain: 1, meal_related_pain: 1, hypertension: 1, hyperlipidemia: 0, diabetes: 1, anxiety_depression: 0, prior_abdominal_surgery: 1, gallbladder_wall_thickening: 1, max_stone_diameter_mm: 18, stone_number: "multiple", stone_location: "distal_common_bile_duct", common_bile_duct_diameter_mm: 10, fatty_liver: 0, gallbladder_atrophy: 0, alt: 260, ast: 210, alp: 240, ggt: 520, total_bilirubin: 62, total_bile_acid: 110, total_cholesterol: 5.1, triglyceride: 1.7, alpha_fetoprotein: 4.8, ca199: 180 } },
  ];

  function t(key) {
    return uiText[language][key];
  }

  function sampleLabel(sample) {
    return sample.label[language];
  }

  function fieldLabel(field) {
    if (language === "zh") return field.label;
    const match = field.label.match(/\(([^)]+)\)\s*$/);
    return match ? match[1] : field.name.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function optionLabel(option) {
    const key = String(option.value);
    const label = valueLabels[language][key] || option.label || key;
    return `${label} (${option.value})`;
  }

  function renderSamples(selectedId = sampleSelect.value) {
    sampleSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = t("samplePlaceholder");
    sampleSelect.appendChild(placeholder);

    for (const sample of sampleProfiles) {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sampleLabel(sample);
      sampleSelect.appendChild(option);
    }
    sampleSelect.value = selectedId;
    updateSampleBadge();
  }

  function renderForm(values = readInputsOrDefaults()) {
    form.innerHTML = "";
    for (const field of model.fields) {
      const wrapper = document.createElement("div");
      wrapper.className = "field";

      const label = document.createElement("label");
      label.htmlFor = field.name;
      label.textContent = fieldLabel(field);
      wrapper.appendChild(label);

      if (field.kind === "number") {
        const control = document.createElement("div");
        control.className = "number-control";

        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.id = field.name;
        input.name = field.name;
        input.value = values[field.name] ?? field.default ?? "";
        input.dataset.numberInput = field.name;
        control.appendChild(input);

        if (Number.isFinite(Number(field.min)) && Number.isFinite(Number(field.max)) && Number(field.max) > Number(field.min)) {
          const slider = document.createElement("input");
          slider.type = "range";
          slider.min = field.min;
          slider.max = field.max;
          slider.step = rangeStep(field.min, field.max);
          slider.value = clamp(Number(input.value), Number(field.min), Number(field.max));
          slider.dataset.rangeFor = field.name;
          control.appendChild(slider);
        }

        wrapper.appendChild(control);

        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = `${t("rangeHint")} ${formatNumber(field.min)} - ${formatNumber(field.max)}`;
        wrapper.appendChild(hint);
      } else {
        const select = document.createElement("select");
        select.id = field.name;
        select.name = field.name;
        for (const option of field.options) {
          const node = document.createElement("option");
          node.value = option.value;
          node.textContent = optionLabel(option);
          select.appendChild(node);
        }
        select.value = values[field.name] ?? field.default;
        wrapper.appendChild(select);

        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = t("selectHint");
        wrapper.appendChild(hint);
      }
      form.appendChild(wrapper);
    }
  }

  function updateStaticText() {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = t("pageTitle");
    document.getElementById("title").textContent = t("title");
    document.getElementById("subtitle").textContent = t("subtitle");
    document.getElementById("riskLabel").textContent = t("riskLabel");
    document.getElementById("modelDetailsSummary").textContent = t("modelDetails");
    document.getElementById("metricModelLabel").textContent = t("model");
    document.getElementById("metricAucLabel").textContent = t("auc");
    document.getElementById("metricSampleLabel").textContent = t("sampleSize");
    document.getElementById("noticeText").textContent = t("notice");
    document.getElementById("sampleCaption").textContent = t("sampleCaption");
    document.getElementById("resetForm").textContent = t("reset");
    document.getElementById("predict").textContent = t("predict");
    languageButtons.forEach((button) => button.classList.toggle("active", button.dataset.lang === language));
    if (lastProbability === null) riskBand.textContent = t("initialRisk");
  }

  function readInputsOrDefaults() {
    if (!form.elements.length) {
      return Object.fromEntries(model.fields.map((field) => [field.name, field.default]));
    }
    return readInputs();
  }

  function rangeStep(min, max) {
    const span = Math.abs(Number(max) - Number(min));
    if (span >= 100) return 1;
    if (span >= 10) return 0.1;
    return 0.01;
  }

  function clamp(value, min, max) {
    if (!Number.isFinite(value)) return min;
    return Math.min(max, Math.max(min, value));
  }

  function formatNumber(value) {
    if (value === null || Number.isNaN(Number(value))) return "--";
    const number = Number(value);
    return Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(2);
  }

  function formatInputNumber(value, step) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (Number(step) >= 1) return String(Math.round(number));
    if (Number(step) >= 0.1) return number.toFixed(1);
    return number.toFixed(2);
  }

  function readInputs() {
    const input = {};
    for (const field of model.fields) {
      const element = form.elements[field.name];
      if (field.kind === "number") {
        const value = Number(element.value);
        input[field.name] = Number.isFinite(value) ? value : null;
      } else {
        input[field.name] = numericIfPossible(element.value);
      }
    }
    return input;
  }

  function numericIfPossible(value) {
    if (value === "0" || value === "1") return Number(value);
    return value;
  }

  function transform(input) {
    const output = [];
    const numeric = model.preprocessing.numeric;
    numeric.columns.forEach((column, index) => {
      let value = Number(input[column]);
      if (!Number.isFinite(value)) value = numeric.median[index];
      output.push((value - numeric.mean[index]) / numeric.scale[index]);
    });

    const categorical = model.preprocessing.categorical;
    categorical.columns.forEach((column, index) => {
      let value = input[column];
      if (value === null || value === undefined || value === "") value = categorical.fill[index];
      for (const category of categorical.categories[index]) output.push(String(value) === String(category) ? 1 : 0);
    });
    return output;
  }

  function predictTree(tree, features) {
    let node = 0;
    while (tree.feature[node] >= 0) {
      node = features[tree.feature[node]] <= tree.threshold[node] ? tree.childrenLeft[node] : tree.childrenRight[node];
    }
    return tree.probability[node];
  }

  function predictProbability(input) {
    const features = transform(input);
    let total = 0;
    for (const tree of model.forest.trees) total += predictTree(tree, features);
    return total / model.forest.trees.length;
  }

  function renderRisk(probability) {
    lastProbability = probability;
    const percent = probability * 100;
    riskValue.textContent = `${percent.toFixed(1)}%`;
    riskBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;

    if (probability >= 0.5) {
      riskBar.style.backgroundColor = "var(--rose)";
      riskBand.textContent = t("highRisk");
    } else if (probability >= 0.2) {
      riskBar.style.backgroundColor = "var(--amber)";
      riskBand.textContent = t("midRisk");
    } else {
      riskBar.style.backgroundColor = "var(--green)";
      riskBand.textContent = t("lowRisk");
    }
  }

  function updateRisk() {
    renderRisk(predictProbability(readInputs()));
  }

  function setFormValues(values) {
    for (const field of model.fields) {
      const element = form.elements[field.name];
      if (!element) continue;
      const value = Object.prototype.hasOwnProperty.call(values, field.name) ? values[field.name] : field.default;
      element.value = value ?? "";
      syncRangeFromInput(field.name);
    }
  }

  function applySample(id) {
    const sample = sampleProfiles.find((item) => item.id === id);
    if (!sample) return;
    setFormValues(sample.values);
    sampleSelect.value = id;
    updateSampleBadge();
    updateRisk();
  }

  function updateSampleBadge() {
    const sample = sampleProfiles.find((item) => item.id === sampleSelect.value);
    currentSample.textContent = sample ? sampleLabel(sample) : t("customInput");
    currentSample.dataset.state = sample ? "sample" : "custom";
  }

  function syncRangeFromInput(name) {
    const input = form.elements[name];
    const range = form.querySelector(`[data-range-for="${name}"]`);
    if (!input || !range) return;
    const number = Number(input.value);
    if (Number.isFinite(number)) range.value = clamp(number, Number(range.min), Number(range.max));
  }

  function syncInputFromRange(range) {
    const input = form.elements[range.dataset.rangeFor];
    if (!input) return;
    input.value = formatInputNumber(range.value, range.step);
  }

  function markCustomInput() {
    sampleSelect.value = "";
    updateSampleBadge();
  }

  function resetForm() {
    sampleSelect.value = "";
    updateSampleBadge();
    const defaults = Object.fromEntries(model.fields.map((field) => [field.name, field.default]));
    setFormValues(defaults);
    lastProbability = null;
    riskValue.textContent = "--";
    riskBar.style.width = "0";
    riskBar.style.backgroundColor = "var(--green)";
    riskBand.textContent = t("initialRisk");
  }

  function setLanguage(nextLanguage) {
    if (nextLanguage === language) return;
    const values = readInputsOrDefaults();
    const selectedSample = sampleSelect.value;
    language = nextLanguage;
    updateStaticText();
    renderSamples(selectedSample);
    renderForm(values);
    if (lastProbability !== null) renderRisk(lastProbability);
  }

  updateStaticText();
  renderSamples();
  renderForm();

  languageButtons.forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.lang)));
  sampleSelect.addEventListener("change", () => applySample(sampleSelect.value));
  form.addEventListener("input", (event) => {
    const target = event.target;
    if (target.dataset.rangeFor) {
      syncInputFromRange(target);
    } else if (target.dataset.numberInput) {
      syncRangeFromInput(target.dataset.numberInput);
    }
    markCustomInput();
    updateRisk();
  });
  form.addEventListener("change", () => {
    markCustomInput();
    updateRisk();
  });
  document.getElementById("predict").addEventListener("click", updateRisk);
  document.getElementById("resetForm").addEventListener("click", resetForm);
})();