function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();

            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

let chart;

async function obtenerDatos() {
    const from = document.getElementById('from').value;
    const to = document.getElementById('to').value;
    const brand = document.getElementById('brand').value;

    const csrftoken = getCookie('csrftoken');

    try {
    const response = await fetch("/getReturns", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': csrftoken
        },
        body: new URLSearchParams({
            'from': from,
            'to': to,
            'brand': brand
        })
    });

        const data = await response.json();
        console.log('Response data:', data); // Para depuración

        if (response.ok) {
        graficarDatos(data);
    } else {
            // Show error message to user
            document.getElementById("beforeText").style.display = "block";
            document.getElementById("afterText").style.display = "none";
            document.getElementById("analysisResult").style.display = "none";
            document.getElementById("beforeText").innerHTML = `<h4 class="text-danger">Error: ${data.error || 'Error desconocido'}</h4>`;
            
            // Clear any existing charts
            if (chart) {
                chart.destroy();
            }
        }
    } catch (error) {
        console.error('Error completo:', error); // Para depuración
        document.getElementById("beforeText").style.display = "block";
        document.getElementById("afterText").style.display = "none";
        document.getElementById("analysisResult").style.display = "none";
        document.getElementById("beforeText").innerHTML = '<h4 class="text-danger">Error al conectar con el servidor</h4>';
    }
}

function graficarDatos(data) {
    document.getElementById("beforeText").style.display = "none"
    document.getElementById("afterText").style.display = "block"
    document.getElementById("analysisResult").style.display = "block"
    document.getElementById("brandId").innerHTML = data.brand
    
    // Gráfico principal (precios y SMA_5)
    const labels = data.data.map(item => item.date);
    const closingPrices = data.data.map(item => item.close);
    const smaValues = data.data.map(item => item.sma_5);
    const ctx = document.getElementById('chart').getContext('2d');

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: `Precio de Cierre de ${data.brand}`,
                    data: closingPrices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    fill: false,
                },
                {
                    label: `Media Móvil Simple (SMA_5) de ${data.brand}`,
                    data: smaValues,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Precio (USD)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (tooltipItem) {
                            return `${tooltipItem.dataset.label}: $${tooltipItem.raw.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });

    // Renderizar el análisis textual
    document.getElementById("analysisResultText").innerHTML = data.analysis.text;
    
    // Crear los gráficos de análisis estadístico
    createStatisticalCharts(data.analysis);
}

function createStatisticalCharts(analysisData) {
    // Limpiar gráficos anteriores
    const chartIds = [
        'priceDistributionChart', 
        'movingAveragesChart', 
        'volatilityChart', 
        'bollingerChart', 
        'volumeChart',
        'rsiChart',
        'macdChart',
        'momentumChart',
        'predictionChart'
    ];
    
    chartIds.forEach(id => {
        const chartContainer = document.getElementById(id);
        if (chartContainer) {
            const parent = chartContainer.parentNode;
            parent.removeChild(chartContainer);
            
            // Crear nuevo canvas
            const newCanvas = document.createElement('canvas');
            newCanvas.id = id;
            document.querySelector('.analysis-charts').appendChild(newCanvas);
        }
    });

    // 1. Gráfico de Precios y Medias Móviles
    createPriceChart(analysisData);
    
    // 2. Gráfico de Bandas de Bollinger
    createBollingerChart(analysisData);
    
    // 3. Gráfico de RSI
    createRSIChart(analysisData);
    
    // 4. Gráfico de MACD
    createMACDChart(analysisData);
    
    // 5. Gráfico de Volatilidad
    createVolatilityChart(analysisData);
    
    // 6. Gráfico de Volumen
    createVolumeChart(analysisData);
    
    // 7. Gráfico de Momentum
    createMomentumChart(analysisData);
    
    // 8. Gráfico de Predicción
    createPredictionChart(analysisData);
    
    // 9. Gráfico de Distribución de Retornos
    createReturnsDistributionChart(analysisData);
}

function createPriceChart(analysisData) {
    const ctx = document.getElementById('movingAveragesChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.moving_averages.dates,
            datasets: [
                {
                    label: 'Precio de Cierre',
                    data: analysisData.price_data.close,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'SMA 20',
                    data: analysisData.technical_indicators.moving_averages.sma_20,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'SMA 50',
                    data: analysisData.technical_indicators.moving_averages.sma_50,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Precios y Medias Móviles'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precio (USD)'
                    }
                }
            }
        }
    });
}

function createBollingerChart(analysisData) {
    const ctx = document.getElementById('bollingerChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.bollinger_bands.dates,
            datasets: [
                {
                    label: 'Precio',
                    data: analysisData.price_data.close,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Banda Superior',
                    data: analysisData.technical_indicators.bollinger_bands.upper,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'Banda Media',
                    data: analysisData.technical_indicators.bollinger_bands.middle,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'Banda Inferior',
                    data: analysisData.technical_indicators.bollinger_bands.lower,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Bandas de Bollinger'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precio (USD)'
                    }
                }
            }
        }
    });
}

function createRSIChart(analysisData) {
    const ctx = document.getElementById('rsiChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.rsi.dates,
            datasets: [{
                label: 'RSI (14)',
                data: analysisData.technical_indicators.rsi.values,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'RSI (Relative Strength Index)'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'RSI'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

function createMACDChart(analysisData) {
    const ctx = document.getElementById('macdChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.macd.dates,
            datasets: [
                {
                    label: 'MACD',
                    data: analysisData.technical_indicators.macd.macd,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Señal',
                    data: analysisData.technical_indicators.macd.signal,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'Histograma',
                    data: analysisData.technical_indicators.macd.histogram,
                    type: 'bar',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'MACD (Moving Average Convergence Divergence)'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor'
                    }
                }
            }
        }
    });
}

function createVolatilityChart(analysisData) {
    const ctx = document.getElementById('volatilityChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.volatility.dates,
            datasets: [
                {
                    label: 'Volatilidad',
                    data: analysisData.technical_indicators.volatility.values,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'ATR',
                    data: analysisData.technical_indicators.volatility.atr,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Volatilidad y ATR'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor'
                    }
                }
            }
        }
    });
}

function createVolumeChart(analysisData) {
    const ctx = document.getElementById('volumeChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: analysisData.technical_indicators.volume_analysis.dates,
            datasets: [
                {
                    label: 'Volumen',
                    data: analysisData.technical_indicators.volume_analysis.volume,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Volumen SMA',
                    data: analysisData.technical_indicators.volume_analysis.volume_sma,
                    type: 'line',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Análisis de Volumen'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Volumen'
                    }
                }
            }
        }
    });
}

function createMomentumChart(analysisData) {
    const ctx = document.getElementById('momentumChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.momentum.dates,
            datasets: [
                {
                    label: 'Momentum',
                    data: analysisData.technical_indicators.momentum.values,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Rate of Change',
                    data: analysisData.technical_indicators.momentum.roc,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Indicadores de Momentum'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor'
                    }
                }
            }
        }
    });
}

function createPredictionChart(analysisData) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: analysisData.technical_indicators.prediction.dates,
            datasets: [
                {
                    label: 'Precio Real',
                    data: analysisData.price_data.close,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Predicción Lineal',
                    data: analysisData.technical_indicators.prediction.linear,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Predicción de Precios'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precio (USD)'
                    }
                }
            }
        }
    });
}

function createReturnsDistributionChart(analysisData) {
    const ctx = document.getElementById('priceDistributionChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: analysisData.statistical_analysis.returns_distribution.bins,
            datasets: [{
                label: 'Distribución de Retornos',
                data: analysisData.statistical_analysis.returns_distribution.values,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Distribución de Retornos Diarios'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Retorno (%)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frecuencia'
                    }
                }
            }
        }
    });
}