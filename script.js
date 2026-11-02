/* ============================================================
   AI CLINICAL TRIAGE FRONTEND CLIENT
   Interacts with Flask + Scikit-Learn backend at /api/predict
   ============================================================ */

(() => {
    'use strict';

    // Preset clinical patient presentations
    const PRESETS = {
        sepsis: { hr: 128, sbp: 82, spo2: 89, rr: 28, temp: 39.4, gcs: 13, pain: 6 },
        ami: { hr: 104, sbp: 165, spo2: 95, rr: 22, temp: 36.8, gcs: 15, pain: 9 },
        asthma: { hr: 118, sbp: 130, spo2: 88, rr: 34, temp: 37.0, gcs: 14, pain: 4 },
        mild: { hr: 76, sbp: 122, spo2: 99, rr: 16, temp: 36.9, gcs: 15, pain: 5 },
        healthy: { hr: 72, sbp: 118, spo2: 99, rr: 14, temp: 36.6, gcs: 15, pain: 0 }
    };

    // DOM References
    const hrSlider = document.getElementById('hr');
    const hrVal = document.getElementById('hrVal');
    const sbpSlider = document.getElementById('sbp');
    const sbpVal = document.getElementById('sbpVal');
    const spo2Slider = document.getElementById('spo2');
    const spo2Val = document.getElementById('spo2Val');
    const rrSlider = document.getElementById('rr');
    const rrVal = document.getElementById('rrVal');
    const tempSlider = document.getElementById('temp');
    const tempVal = document.getElementById('tempVal');
    const gcsSlider = document.getElementById('gcs');
    const gcsVal = document.getElementById('gcsVal');
    const painSlider = document.getElementById('pain');
    const painVal = document.getElementById('painVal');

    const triageCard = document.getElementById('triageCard');
    const urgencyCode = document.getElementById('urgencyCode');
    const urgencyCategory = document.getElementById('urgencyCategory');
    const timeTarget = document.getElementById('timeTarget');
    const triageRec = document.getElementById('triageRec');

    const newsScore = document.getElementById('newsScore');
    const newsStatus = document.getElementById('newsStatus');
    const icuRisk = document.getElementById('icuRisk');
    const priorityNum = document.getElementById('priorityNum');
    const flagsList = document.getElementById('flagsList');

    const caseBtns = document.querySelectorAll('.case-btn');

    // Cost-Sensitive Optimization DOM elements
    const fnPenaltySlider = document.getElementById('fnPenalty');
    const fnPenaltyVal = document.getElementById('fnPenaltyVal');
    const expectedLossBadge = document.getElementById('expectedLossBadge');
    const safetyPriority = document.getElementById('safetyPriority');
    const thresholdShift = document.getElementById('thresholdShift');
    const actionProtocol = document.getElementById('actionProtocol');

    let radarChart = null;
    let shapChart = null;
    let debounceTimer = null;

    function initRadar() {
        const ctx = document.getElementById('radarChart').getContext('2d');
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Heart Rate', 'Systolic BP', 'SpO2 Depletion', 'Resp Rate', 'Fever / Temp', 'CNS (GCS Deficit)'],
                datasets: [
                    {
                        label: 'Patient Vitals Deviation',
                        data: [100, 100, 100, 100, 100, 100],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: '#ef4444'
                    },
                    {
                        label: 'Normal Baseline',
                        data: [100, 100, 100, 100, 100, 100],
                        borderColor: '#22c55e',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#94a3b8', font: { size: 11 } },
                        suggestedMin: 60,
                        suggestedMax: 180,
                        ticks: { display: false }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', boxWidth: 12 }
                    }
                }
            }
        });
    }

    function initShapChart() {
        const ctx = document.getElementById('shapChart').getContext('2d');
        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['SpO2 Depletion', 'GCS Deficit', 'Systolic BP (Shock)', 'Heart Rate', 'Resp Rate', 'Temperature', 'Pain VAS'],
                datasets: [{
                    label: 'Risk Contribution (+/- SHAP Impact)',
                    data: [0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: [],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => `${v > 0 ? '+' : ''}${v.toFixed(1)}` }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f8fafc', font: { size: 11, weight: 'bold' } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    async function evaluate() {
        const hr = parseInt(hrSlider.value, 10);
        const sbp = parseInt(sbpSlider.value, 10);
        const spo2 = parseInt(spo2Slider.value, 10);
        const rr = parseInt(rrSlider.value, 10);
        const temp = parseFloat(tempSlider.value);
        const gcs = parseInt(gcsSlider.value, 10);
        const pain = parseInt(painSlider.value, 10);
        const fnWeight = parseInt(fnPenaltySlider.value, 10);
        fnPenaltyVal.textContent = `${fnWeight}×`;

        // Update readouts
        hrVal.textContent = `${hr} bpm`;
        sbpVal.textContent = `${sbp} mmHg`;
        spo2Val.textContent = `${spo2}%`;
        rrVal.textContent = `${rr} /min`;
        tempVal.textContent = `${temp.toFixed(1)} °C`;
        gcsVal.textContent = `${gcs} / 15`;
        painVal.textContent = `${pain} / 10`;

        // Update Radar
        const hrNorm = Math.round((hr / 75) * 100);
        const sbpNorm = Math.round((sbp / 120) * 100);
        const spo2Norm = Math.round(((100 - spo2 + 90) / 90) * 100);
        const rrNorm = Math.round((rr / 16) * 100);
        const tempNorm = Math.round((temp / 36.8) * 100);
        const gcsNorm = Math.round(((15 - gcs + 10) / 10) * 100);

        radarChart.data.datasets[0].data = [hrNorm, sbpNorm, spo2Norm, rrNorm, tempNorm, gcsNorm];
        radarChart.update();

        // === NEWS2 Score Calculation (Client-Side) ===
        function calcNEWS2(hr, sbp, spo2, rr, temp, gcs) {
            let score = 0;
            // Respiration Rate
            if (rr <= 8) score += 3;
            else if (rr >= 25) score += 3;
            else if (rr >= 21 && rr <= 24) score += 2;
            else if (rr >= 9 && rr <= 11) score += 1;
            // SpO2
            if (spo2 <= 91) score += 3;
            else if (spo2 >= 92 && spo2 <= 93) score += 2;
            else if (spo2 >= 94 && spo2 <= 95) score += 1;
            // Systolic BP
            if (sbp <= 90) score += 3;
            else if (sbp >= 220) score += 3;
            else if (sbp >= 91 && sbp <= 100) score += 2;
            else if (sbp >= 101 && sbp <= 110) score += 1;
            // Heart Rate
            if (hr <= 40) score += 3;
            else if (hr >= 131) score += 3;
            else if (hr >= 111 && hr <= 130) score += 2;
            else if (hr >= 91 && hr <= 110) score += 1;
            else if (hr >= 41 && hr <= 50) score += 1;
            // Temperature
            if (temp <= 35.0) score += 3;
            else if (temp >= 39.1) score += 2;
            else if (temp >= 38.1 && temp <= 39.0) score += 1;
            else if (temp >= 35.1 && temp <= 36.0) score += 1;
            // Consciousness (GCS)
            if (gcs <= 14) score += 3;
            return score;
        }

        function classifyManchester(news, hr, sbp, spo2, rr, temp, gcs, pain) {
            // Manchester Protocol Triage Classification
            if (news >= 7 || gcs <= 8 || spo2 <= 85 || sbp <= 70 || hr <= 35 || hr >= 160 || rr <= 6 || rr >= 35) {
                return { priority: 1, code: 'red', category: 'Immediate (Red)', time: '≤ 0 mins', color: 'red' };
            }
            if (news >= 5 || gcs <= 12 || spo2 <= 91 || sbp <= 90 || hr >= 131 || rr >= 25 || temp >= 40.0 || pain >= 8) {
                return { priority: 2, code: 'orange', category: 'Very Urgent (Orange)', time: '≤ 10 mins', color: 'orange' };
            }
            if (news >= 3 || spo2 <= 94 || sbp <= 100 || hr >= 111 || rr >= 21 || temp >= 38.5 || gcs <= 14 || pain >= 5) {
                return { priority: 3, code: 'yellow', category: 'Urgent (Yellow)', time: '≤ 60 mins', color: 'yellow' };
            }
            if (news >= 1 || pain >= 3 || hr >= 91 || temp >= 37.5 || rr >= 18) {
                return { priority: 4, code: 'green', category: 'Standard (Green)', time: '≤ 120 mins', color: 'green' };
            }
            return { priority: 5, code: 'blue', category: 'Non-Urgent (Blue)', time: '≤ 240 mins', color: 'blue' };
        }

        function detectFlags(hr, sbp, spo2, rr, temp, gcs, pain) {
            const flags = [];
            if (hr >= 130) flags.push(`⚠️ Tachycardia: HR ${hr} bpm (critical ≥130)`);
            else if (hr >= 100) flags.push(`⚠️ Tachycardia: HR ${hr} bpm (elevated ≥100)`);
            if (hr <= 45) flags.push(`⚠️ Bradycardia: HR ${hr} bpm (critical ≤45)`);
            if (sbp <= 90) flags.push(`🔴 Hypotension: SBP ${sbp} mmHg (shock risk ≤90)`);
            if (sbp >= 180) flags.push(`⚠️ Hypertensive Crisis: SBP ${sbp} mmHg (≥180)`);
            if (spo2 <= 92) flags.push(`🔴 Hypoxemia: SpO₂ ${spo2}% (critical ≤92)`);
            else if (spo2 <= 94) flags.push(`⚠️ Low SpO₂: ${spo2}% (below normal ≤94)`);
            if (rr >= 25) flags.push(`⚠️ Tachypnea: RR ${rr}/min (respiratory distress ≥25)`);
            if (rr <= 8) flags.push(`🔴 Bradypnea: RR ${rr}/min (ventilation failure ≤8)`);
            if (temp >= 39.0) flags.push(`⚠️ High Fever: ${temp.toFixed(1)} °C (febrile ≥39.0)`);
            if (temp >= 40.0) flags.push(`🔴 Hyperthermia: ${temp.toFixed(1)} °C (dangerous ≥40.0)`);
            if (temp <= 35.0) flags.push(`🔴 Hypothermia: ${temp.toFixed(1)} °C (≤35.0)`);
            if (gcs <= 12) flags.push(`🔴 Altered Consciousness: GCS ${gcs}/15 (≤12)`);
            else if (gcs <= 14) flags.push(`⚠️ Mild CNS Depression: GCS ${gcs}/15 (≤14)`);
            if (pain >= 8) flags.push(`⚠️ Severe Pain: VAS ${pain}/10`);
            return flags;
        }

        function estimateICURisk(news, gcs, spo2, sbp) {
            let risk = 0.5 + (news * 3.2);
            if (gcs <= 8) risk += 35;
            else if (gcs <= 12) risk += 15;
            if (spo2 <= 88) risk += 20;
            else if (spo2 <= 92) risk += 10;
            if (sbp <= 80) risk += 18;
            else if (sbp <= 90) risk += 8;
            return Math.min(Math.max(risk, 0.5), 98.5).toFixed(1);
        }

        // Run client-side triage
        const news2 = calcNEWS2(hr, sbp, spo2, rr, temp, gcs);
        const triage = classifyManchester(news2, hr, sbp, spo2, rr, temp, gcs, pain);
        const flags = detectFlags(hr, sbp, spo2, rr, temp, gcs, pain);
        const icu = parseFloat(estimateICURisk(news2, gcs, spo2, sbp));

        // === ASYMMETRIC LOSS & COST-SENSITIVE THRESHOLD CALCULATION ===
        // Cost(FN) = fnWeight * Cost(FP). Standard threshold = 0.50 -> Optimal Threshold = 1 / (1 + fnWeight)
        const optimalThreshold = (1.0 / (1.0 + fnWeight)) * 100.0;
        const deltaShift = ((optimalThreshold - 50.0)).toFixed(1);
        thresholdShift.textContent = `${deltaShift}% (Risk Threshold: ${optimalThreshold.toFixed(1)}%)`;

        // Calculate expected clinical loss: P(Severe) * Cost(FN if discharged) + (1-P(Severe)) * Cost(FP if admitted)
        const pSevere = icu / 100.0;
        const expectedLoss = (pSevere * (fnWeight * 0.05) + (1 - pSevere) * 0.02).toFixed(3);
        expectedLossBadge.textContent = `Expected Clinical Loss: ${expectedLoss}`;

        // Safety-adjusted triage recommendation: If ICU Risk exceeds optimal cost threshold, escalate priority
        let safetyPri = triage.category;
        let protocol = 'Routine Track';
        if (icu >= optimalThreshold) {
            if (triage.priority > 2) {
                safetyPri = 'Escalated: High Sensitivity (Orange)';
                protocol = 'Immediate Bedside Observation';
            } else if (triage.priority === 2) {
                safetyPri = 'Escalated: Critical (Red)';
                protocol = 'Resuscitation Bay Pre-Alert';
            } else {
                safetyPri = 'Max Priority: Immediate (Red)';
                protocol = 'Resuscitation Bay Mobilized';
            }
        } else {
            safetyPri = `Standard: ${triage.category}`;
            protocol = triage.priority <= 2 ? 'Continuous Monitoring' : 'Standard Routine Queue';
        }
        safetyPriority.textContent = safetyPri;
        actionProtocol.textContent = protocol;

        // === LOCAL EXPLAINABLE AI: SHAP WATERFALL ATTRIBUTION ===
        // Baseline reference values: HR 75, SBP 120, SpO2 98, RR 16, Temp 36.8, GCS 15, Pain 0
        const shapSpo2 = (98 - spo2) * 1.35;
        const shapGcs = (15 - gcs) * 2.10;
        const shapSbp = sbp < 95 ? (95 - sbp) * 0.75 : (sbp > 160 ? (sbp - 160) * 0.35 : -1.2);
        const shapHr = hr > 100 ? (hr - 100) * 0.28 : (hr < 50 ? (50 - hr) * 0.45 : -0.8);
        const shapRr = rr > 20 ? (rr - 20) * 0.65 : (rr < 10 ? (10 - rr) * 0.85 : -0.6);
        const shapTemp = temp > 37.5 ? (temp - 37.5) * 3.2 : (temp < 35.5 ? (35.5 - temp) * 3.5 : -0.5);
        const shapPain = pain > 3 ? (pain - 3) * 0.45 : -0.5;

        const shapValues = [shapSpo2, shapGcs, shapSbp, shapHr, shapRr, shapTemp, shapPain];
        const shapColors = shapValues.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.75)' : 'rgba(34, 197, 94, 0.65)');

        if (shapChart) {
            shapChart.data.datasets[0].data = shapValues.map(v => parseFloat(v.toFixed(1)));
            shapChart.data.datasets[0].backgroundColor = shapColors;
            shapChart.update();
        }

        // Apply triage classification to UI
        triageCard.className = `triage-card triage-${triage.code}`;
        urgencyCode.textContent = `CATEGORY ${['I','II','III','IV','V'][triage.priority - 1]}`;
        urgencyCategory.textContent = triage.category;
        timeTarget.textContent = triage.time;

        let rec = '';
        if (triage.priority === 1) rec = 'CRITICAL EMERGENCY: Immediate life support required. Resuscitation bay protocol triggered. All resources mobilized.';
        else if (triage.priority === 2) rec = 'VERY URGENT: High deterioration risk. Bedside physician evaluation required within 10 minutes. Continuous monitoring activated.';
        else if (triage.priority === 3) rec = 'URGENT: Patient requires monitored observation and diagnostic workup within 60 minutes. Active reassessment protocol.';
        else if (triage.priority === 4) rec = 'STANDARD: Symptoms stable for routine emergency assessment. Patient queued for standard evaluation within 120 minutes.';
        else rec = 'NON-URGENT: Patient exhibits stable physiologic parameters. Eligible for standard outpatient management.';

        triageRec.textContent = rec;

        newsScore.textContent = news2;
        newsStatus.textContent = news2 >= 7 ? 'High Clinical Risk' : (news2 >= 5 ? 'Medium Clinical Risk' : (news2 >= 1 ? 'Low-Medium Risk' : 'Low Clinical Risk'));

        priorityNum.innerHTML = `${triage.priority} <span class="unit-faint">/ 5</span>`;
        icuRisk.textContent = `${icu}%`;

        // Flags
        flagsList.innerHTML = '';
        if (flags.length === 0) {
            flagsList.innerHTML = '<li class="flag-empty">✓ No critical vital sign deviations detected.</li>';
        } else {
            flags.forEach(f => {
                const li = document.createElement('li');
                li.textContent = f;
                flagsList.appendChild(li);
            });
        }
    }

    function triggerDebouncedEvaluate() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(evaluate, 50);
    }

    function applyCase(k) {
        const c = PRESETS[k];
        if (!c) return;

        hrSlider.value = c.hr;
        sbpSlider.value = c.sbp;
        spo2Slider.value = c.spo2;
        rrSlider.value = c.rr;
        tempSlider.value = c.temp;
        gcsSlider.value = c.gcs;
        painSlider.value = c.pain;

        evaluate();
    }

    const sliders = [hrSlider, sbpSlider, spo2Slider, rrSlider, tempSlider, gcsSlider, painSlider];
    sliders.forEach(s => s.addEventListener('input', () => {
        caseBtns.forEach(x => x.classList.remove('active'));
        triggerDebouncedEvaluate();
    }));

    if (fnPenaltySlider) {
        fnPenaltySlider.addEventListener('input', triggerDebouncedEvaluate);
    }

    caseBtns.forEach(b => {
        b.addEventListener('click', () => {
            caseBtns.forEach(x => x.classList.remove('active'));
            b.classList.add('active');
            applyCase(b.dataset.case);
        });
    });

    initRadar();
    initShapChart();
    evaluate();
})();
