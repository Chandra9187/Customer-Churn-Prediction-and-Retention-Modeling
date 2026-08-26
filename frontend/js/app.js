const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : ''; // Empty string means use relative path when hosted on same origin

document.addEventListener('DOMContentLoaded', () => {
    // --- Setup UI Handlers ---
    const satisfactionSlider = document.getElementById('satisfaction_score');
    const satisfactionVal = document.getElementById('satisfaction_val');

    if (satisfactionSlider && satisfactionVal) {
        satisfactionSlider.addEventListener('input', (e) => {
            satisfactionVal.textContent = e.target.value;
        });
    }

    // Smooth scroll for nav links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });

            // Update active state
            document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // --- Fetch Metrics on Load ---
    async function loadMetrics() {
        const metricsContainer = document.getElementById('metrics-container');
        try {
            const response = await fetch(`${API_BASE_URL}/metrics`);
            if (!response.ok) throw new Error("Failed to load metrics");

            const data = await response.json();
            renderMetrics(data.models, metricsContainer);

            // Update hero ROC AUC based on actual best model
            const bestModel = data.models["XGBoost (Optimized)"] || data.models["XGBoost"];
            if (bestModel && bestModel.roc_auc) {
                document.getElementById('hero-roc-auc').textContent = bestModel.roc_auc.toFixed(2);
            }
        } catch (error) {
            console.error("Error fetching metrics:", error);
            // Fallback skeleton replacement if API is down
            fallbackRenderMetrics(metricsContainer);
        }
    }

    function renderMetrics(models, container) {
        container.innerHTML = ''; // Clear skeletons

        Object.entries(models).forEach(([name, metrics]) => {
            const isHighlight = name.includes('XGBoost');
            const card = document.createElement('div');
            card.className = `model-card glass-panel ${isHighlight ? 'highlight' : ''}`;

            card.innerHTML = `
                <div class="model-name">${name}</div>
                <div class="metric-row">
                    <span>ROC-AUC</span>
                    <span class="val" style="color: ${isHighlight ? '#a78bfa' : '#ffffff'}">${metrics.roc_auc.toFixed(4)}</span>
                </div>
                <div class="metric-row">
                    <span>Accuracy</span>
                    <span class="val">${(metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="metric-row">
                    <span>Recall (Churn Found)</span>
                    <span class="val">${(metrics.recall * 100).toFixed(2)}%</span>
                </div>
                <div class="metric-row">
                    <span>F1 Score</span>
                    <span class="val">${metrics.f1.toFixed(4)}</span>
                </div>
            `;
            container.appendChild(card);
        });
    }

    function fallbackRenderMetrics(container) {
        // Just hardcode the values we know we got during training
        const fallbackData = {
            "Logistic Regression": { roc_auc: 0.9708, accuracy: 0.9353, recall: 0.8404, f1: 0.7936 },
            "Random Forest": { roc_auc: 0.9787, accuracy: 0.9393, recall: 0.9305, f1: 0.8196 },
            "XGBoost (Optimized)": { roc_auc: 0.9787, accuracy: 0.9281, recall: 0.9696, f1: 0.7997 }
        };
        renderMetrics(fallbackData, container);
    }

    // --- Fetch Summary on Load ---
    async function loadSummary() {
        try {
            const response = await fetch(`${API_BASE_URL}/eda/summary`);
            if (!response.ok) throw new Error("Failed to load summary");

            const data = await response.json();

            // Format currency
            const formatter = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            });

            document.getElementById('total-loss-stat').textContent = formatter.format(data.total_expected_loss);
            document.getElementById('top10-loss-stat').textContent = formatter.format(data.top_10_percent_loss);

            const pctText = `${data.top_10_contribution_pct.toFixed(1)}%`;
            document.getElementById('top10-pct-text').textContent = pctText;
            document.getElementById('hero-loss-pct').textContent = pctText;
            document.getElementById('top10-pct-bar').style.width = pctText;

        } catch (error) {
            console.error("Error fetching summary:", error);
            // Keep hardcoded fallback values in HTML if API is down
        }
    }

    // --- Prediction Form Submission ---
    const predictForm = document.getElementById('prediction-form');
    let gaugeChart = null;

    function renderRiskGauge(probability) {
        const ctx = document.getElementById('riskGauge');
        if (!ctx) return;

        if (gaugeChart) {
            gaugeChart.destroy();
        }

        // Convert to percentage
        const p = Math.round(probability * 100);
        let color = '#10b981'; // Green
        if (p > 30) color = '#f59e0b'; // Yellow
        if (p > 70) color = '#ef4444'; // Red

        gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [p, 100 - p],
                    backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270,
                    cutout: '80%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: {
                    animateRotate: true,
                    duration: 1500,
                    easing: 'easeOutQuart'
                }
            }
        });

        document.getElementById('churn-prob-value').textContent = `${p}%`;
        document.getElementById('churn-prob-value').style.color = color;
    }

    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('predict-btn');
        const btnText = btn.querySelector('.btn-text');
        const loader = btn.querySelector('.loader');

        // Show loading state
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        btn.disabled = true;

        // Collect form data
        const formData = new FormData(predictForm);
        const data = {
            credit_score: parseInt(formData.get('credit_score')),
            geography: formData.get('geography'),
            gender: formData.get('gender'),
            age: parseInt(formData.get('age')),
            tenure: parseInt(formData.get('tenure')),
            balance: parseFloat(formData.get('balance')),
            num_of_products: parseInt(formData.get('num_of_products')),
            has_cr_card: document.getElementById('has_cr_card').checked ? 1 : 0,
            is_active_member: document.getElementById('is_active_member').checked ? 1 : 0,
            estimated_salary: parseFloat(formData.get('estimated_salary')),
            complain: document.getElementById('complain').checked ? 1 : 0,
            satisfaction_score: parseInt(formData.get('satisfaction_score')),
            card_type: formData.get('card_type'),
            points_earned: parseInt(formData.get('points_earned'))
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error("Prediction API Failed");

            const result = await response.json();

            // Switch UI states
            document.getElementById('empty-state').classList.add('hidden');
            document.getElementById('result-state').classList.remove('hidden');

            // Render Gauge
            renderRiskGauge(result.churn_probability);

            // Update Details
            const riskBadge = document.getElementById('risk-level-badge');
            riskBadge.textContent = `${result.risk_level} Risk`;
            riskBadge.className = `badge risk-${result.risk_level.toLowerCase()}`;

            const rec = document.getElementById('recommendation-text');
            if (result.risk_level === 'High') {
                rec.textContent = "Immediate Outreach Required";
                rec.className = "text-danger";
            } else if (result.risk_level === 'Medium') {
                rec.textContent = "Monitor & Consider Promotional Offer";
                rec.className = "text-warning";
            } else {
                rec.textContent = "Standard Retention Flow";
                rec.className = "text-success";
            }

            const formatter = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
            });
            document.getElementById('expected-loss-val').textContent = formatter.format(result.expected_loss);

            // Scroll to results slightly
            if (window.innerWidth < 900) {
                document.querySelector('.result-panel').scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

        } catch (error) {
            console.error(error);
            alert("Error running inference. Please ensure the backend is running.");
        } finally {
            // Restore button
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            btn.disabled = false;
        }
    });

    // Initialize data
    loadMetrics();
    loadSummary();
});
