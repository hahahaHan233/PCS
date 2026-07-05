(function () {
  const model = window.PCS_MODEL;
  const form = document.getElementById("predictorForm");
  const riskValue = document.getElementById("riskValue");
  const riskBar = document.getElementById("riskBar");
  const riskBand = document.getElementById("riskBand");

  function optionLabel(option) {
    return `${option.label} (${option.value})`;
  }

  function renderForm() {
    form.innerHTML = "";
    for (const field of model.fields) {
      const wrapper = document.createElement("div");
      wrapper.className = "field";

      const label = document.createElement("label");
      label.htmlFor = field.name;
      label.textContent = field.label;
      wrapper.appendChild(label);

      if (field.kind === "number") {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.id = field.name;
        input.name = field.name;
        input.value = field.default ?? "";
        wrapper.appendChild(input);

        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = `参考范围 ${formatNumber(field.min)} - ${formatNumber(field.max)}`;
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
        select.value = field.default;
        wrapper.appendChild(select);

        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = "请选择最接近的分类";
        wrapper.appendChild(hint);
      }
      form.appendChild(wrapper);
    }
  }

  function formatNumber(value) {
    if (value === null || Number.isNaN(Number(value))) return "--";
    const number = Number(value);
    return Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(2);
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
      if (value === null || value === undefined || value === "") {
        value = categorical.fill[index];
      }
      for (const category of categorical.categories[index]) {
        output.push(String(value) === String(category) ? 1 : 0);
      }
    });
    return output;
  }

  function predictTree(tree, features) {
    let node = 0;
    while (tree.feature[node] >= 0) {
      const featureIndex = tree.feature[node];
      const threshold = tree.threshold[node];
      node = features[featureIndex] <= threshold ? tree.childrenLeft[node] : tree.childrenRight[node];
    }
    return tree.probability[node];
  }

  function predictProbability(input) {
    const features = transform(input);
    let total = 0;
    for (const tree of model.forest.trees) {
      total += predictTree(tree, features);
    }
    return total / model.forest.trees.length;
  }

  function renderRisk(probability) {
    const percent = probability * 100;
    riskValue.textContent = `${percent.toFixed(1)}%`;
    riskBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;

    if (probability >= 0.5) {
      riskBar.style.backgroundColor = "var(--rose)";
      riskBand.textContent = "高风险：达到当前模型阳性阈值";
    } else if (probability >= 0.2) {
      riskBar.style.backgroundColor = "var(--amber)";
      riskBand.textContent = "中等风险：建议结合临床信息复核";
    } else {
      riskBar.style.backgroundColor = "var(--green)";
      riskBand.textContent = "较低风险：低于当前模型阳性阈值";
    }
  }

  function fillExample() {
    const example = {
      age: 56,
      sex: "female",
      occupation: "mental_labor",
      height_cm: 162,
      weight_kg: 63,
      bmi: 24,
      education_level: "high_school",
      smoking: 0,
      alcohol_use: 0,
      symptom_duration: "medium",
      abdominal_pain: 1,
      pain_frequency: "weekly",
      radiating_pain: 0,
      meal_related_pain: 1,
      hypertension: 0,
      hyperlipidemia: 0,
      diabetes: 0,
      anxiety_depression: 0,
      prior_abdominal_surgery: 0,
      gallbladder_wall_thickening: 1,
      max_stone_diameter_mm: 8,
      stone_number: "multiple",
      stone_location: "gallbladder_neck",
      common_bile_duct_diameter_mm: 7,
      fatty_liver: 0,
      gallbladder_atrophy: 0,
      alt: 65,
      ast: 52,
      alp: 160,
      ggt: 180,
      total_bilirubin: 24,
      total_bile_acid: 35,
      total_cholesterol: 5.2,
      triglyceride: 1.8,
      alpha_fetoprotein: 4.2,
      ca199: 38,
    };
    for (const [name, value] of Object.entries(example)) {
      if (form.elements[name]) form.elements[name].value = value;
    }
    renderRisk(predictProbability(readInputs()));
  }

  function resetForm() {
    for (const field of model.fields) {
      if (form.elements[field.name]) form.elements[field.name].value = field.default ?? "";
    }
    riskValue.textContent = "--";
    riskBar.style.width = "0";
    riskBar.style.backgroundColor = "var(--green)";
    riskBand.textContent = "请填写变量后计算";
  }

  renderForm();
  document.getElementById("predict").addEventListener("click", () => {
    renderRisk(predictProbability(readInputs()));
  });
  document.getElementById("fillExample").addEventListener("click", fillExample);
  document.getElementById("resetForm").addEventListener("click", resetForm);
})();
