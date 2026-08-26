document.addEventListener('DOMContentLoaded', () => {
    // Utility functions for rendering charts
    function renderLorenzCurve() {
        const lorenzCtx = document.getElementById('lorenzChart');
        if (!lorenzCtx) return;

        // Realistic Lorenz curve data based on our metrics (Top 10% -> 81% loss)
        const percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        const perfectEquality = percentiles;
        const actualLorenz = [0, 81, 88, 93, 96, 97.5, 98.5, 99.2, 99.7, 99.9, 100]; // 81% of loss at top 10% risky

        new Chart(lorenzCtx, {
            type: 'line',
            data: {
                labels: percentiles.map(p => `${p}%`),
                datasets: [
                    {
                        label: 'Actual Expected Loss Accumulation',
                        data: actualLorenz,
                        borderColor: '#fca5a5',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Uniform Loss Distribution',
                        data: perfectEquality,
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        elements: { point: { radius: 0 } },
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#9ca3af', font: { family: "'Inter', sans-serif" } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Top ${100 - context.parsed.x}% risky account for ${context.parsed.y}% of loss`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Cumulative % of Customers (sorted by risk)', color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#6b7280' }
                    },
                    y: {
                        title: { display: true, text: 'Cumulative % of Expected Loss', color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#6b7280' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    // Call chart renders
    renderLorenzCurve();
});
