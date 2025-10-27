//  مفتاح API الخاص بموقع OpenWeather
const apiKey = "5559cf41152a0a98168aaf2240a76c59";

//  تعريف العناصر من الـ HTML
const cityInput = document.getElementById("cityInput");
const searchBtn = document.getElementById("searchBtn");
const weatherResult = document.getElementById("weatherResult");
const aiPrediction = document.getElementById("aiPrediction");
const tempChartContainer = document.getElementById("tempChartContainer");
const aiChartContainer = document.getElementById("tempChartContainer");

const ctx = document.getElementById("tempChart").getContext("2d");

//  متغيرات الرسم البياني
let tempChart, aiChart;

//  إخفاء النتائج في البداية
weatherResult.style.display = "none";
tempChartContainer.style.display = "none";
aiChartContainer.style.display = "none";

document.getElementById("aiChartContainer").style.display = "none";

//  عند الضغط على الزر
searchBtn.addEventListener("click", () => {
  const city = cityInput.value.trim();
  if (!city) return alert("من فضلك أدخل اسم المدينة");
  fetchWeather(city);
});

//  دالة جلب بيانات الطقس
async function fetchWeather(city) {
  const url = `https://api.openweathermap.org/data/2.5/forecast?q=${encodeURIComponent(city)}&appid=${apiKey}&units=metric&lang=ar;`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    if (data.cod === "404") return showResult("⚠️ المدينة غير موجودة");

    const current = data.list[0];
    const temps = data.list.slice(0, 8).map((d) => d.main.temp);
    const labels = data.list.slice(0, 8).map((d) => d.dt_txt.split(" ")[1].slice(0, 5));

    // 🧠 التنبؤ بالحرارة باستخدام AI
    const predicted = await trainAndPredict(temps);

    showResult(`
      <p>🌍 المدينة: <strong>${city}</strong></p>
      <p>🌡️ درجة الحرارة الحالية: <strong>${current.main.temp}°C</strong></p>
      <p>☁️ الطقس: <strong>${current.weather[0].description}</strong></p>
      <p>💧 الرطوبة: <strong>${current.main.humidity}%</strong></p>
      <p>💨 الرياح: <strong>${current.wind.speed} m/s</strong></p>
    `);

    aiPrediction.innerHTML = predicted
      ? `🤖 التنبؤ القادم: <strong>${predicted.toFixed(1)}°C</strong>`
      : `⚠️ لم يتمكن الذكاء الاصطناعي من التنبؤ`;

    // 📊 إظهار المخططات
    tempChartContainer.style.display = "block";
    document.getElementById("aiChartContainer").style.display = "block";

    renderChart([...labels, "تنبؤ AI"], [...temps, predicted]);
    renderAIChart(temps, predicted);
  } catch (err) {
    console.error(err);
    showResult("حدث خطأ أثناء جلب البيانات، حاول مجددًا");
  }
}

// 🧾 دالة عرض النتيجة
function showResult(html) {
  weatherResult.innerHTML = html;
  weatherResult.style.display = "block";
}

//  دالة تدريب نموذج الذكاء الاصطناعي
async function trainAndPredict(temps) {
  try {
    if (!window.tf) {
      console.error("TensorFlow.js غير محمل!");
      return null;
    }

    const ws = 4;
    if (temps.length <= ws) return avg(temps);

    //  تجهيز البيانات
    const xs = [],
      ys = [];
    for (let i = 0; i < temps.length - ws; i++) {
      xs.push(temps.slice(i, i + ws));
      ys.push(temps[i + ws]);
    }

    //  تطبيع البيانات بين 0 و1
    const min = Math.min(...temps),
      max = Math.max(...temps),
      range = max - min || 1;
    const norm = (v) => (v - min) / range;
    const xsNorm = xs.map((a) => a.map(norm));
    const ysNorm = ys.map(norm);

    //  إنشاء النموذج
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [ws], units: 16, activation: "relu" }));
    model.add(tf.layers.dense({ units: 8, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });

    //  تدريب النموذج
    const xT = tf.tensor2d(xsNorm);
    const yT = tf.tensor2d(ysNorm, [ysNorm.length, 1]);
    await model.fit(xT, yT, { epochs: 60, batchSize: Math.min(8, xsNorm.length) });

    //  التنبؤ
    const last = temps.slice(-ws).map(norm);
    const pred = model.predict(tf.tensor2d([last]));
    const val = (await pred.array())[0][0] * range + min;

    tf.dispose([xT, yT, pred, model]);
    return isNaN(val) ? null : val;
  } catch (e) {
    console.error("خطأ أثناء التنبؤ:", e);
    return null;
  }
}

//  رسم مخطط الخط لدرجات الحرارة
function renderChart(labels, temps) {
  if (tempChart) tempChart.destroy();
  tempChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "درجة الحرارة °C",
          data: temps,
          borderColor: "#ffcc00",
          borderWidth: 3,
          tension: 0.4,
          fill: true,
          pointBackgroundColor: "white",
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "white" } } },
      scales: {
        x: { ticks: { color: "white" } },
        y: { ticks: { color: "white" } },
      },
    },
  });
}

//  رسم مقارنة بين القيم الحقيقية وتوقع AI
function renderAIChart(real, pred) {
  const aiCtx = document.getElementById("aiChart").getContext("2d");
  if (aiChart) aiChart.destroy();
  aiChart = new Chart(aiCtx, {
    type: "bar",
    data: {
      labels: [...real.map((_, i) => `T${i + 1}`), "تنبؤ AI"],
      datasets: [
        {
          label: "الحرارة الفعلية",
          data: real,
          backgroundColor: "rgba(255,255,255,0.4)",
        },
        {
          label: "توقع AI",
          data: [...Array(real.length).fill(null), pred],
          backgroundColor: "#ffcc00",
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "white" } } },
      scales: {
        x: { ticks: { color: "white" } },
        y: { ticks: { color: "white" } },
      },
    },
  });
}

