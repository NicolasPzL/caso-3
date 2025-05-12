# Caso de Estudio 3: Pruebas de Software - Django y Testeo web

## Descripción del Proyecto

Este proyecto tiene como objetivo proporcionar una experiencia práctica en la generación y ejecución de pruebas de software unitarias e integración, utilizando Pytest y Allure. Se deberá implementar dobles de prueba, gestionar archivos, y aplicar fixtures para el correcto manejo del entorno de pruebas. Ambos serán integrados dentro del framework Django. **Usuario:** clase, **Contraseña:** clase2024.

## Actividad a Realizar

1. **Exploración del aplicativo**: Como nota, este trabajo no tiene una descripción detallada de la funcionalidad del mismo. Debe explorar el aplicativo para conocer la funcionalidad del software entregado.
2. Se requiere que implementes un análisis predictivo para evaluar la conveniencia de comprar o vender acciones, basándote en los precios de cierre de la bolsa de la marca seleccionada. Deberás desarrollar la función `analyze_data(prices)` (ya la función está creada y conectada con el front, debes desarrollar el contenido de la misma), la cual debe realizar un análisis exhaustivo del comportamiento de los precios a lo largo del periodo indicado. Esta función no solo debe ser robusta y eficiente, sino que también debe reflejar tu comprensión profunda en analítica de datos, dado que se trata de estudiantes de Ingeniería de Datos y Software, así como tu habilidad en pruebas de software. El resultado esperado es un informe detallado que incluya un análisis cuantitativo con un mínimo de 150 palabras, en el cual se informe al usuario sobre la recomendación de compra o venta de acciones, fundamentada en el comportamiento histórico de los precios de cierre. Asegúrate de utilizar técnicas analíticas adecuadas y presentar los resultados de manera clara y comprensible. El resultado del informe se debe mostrar en el front.
3. **Generación de Pruebas**:
   Según su exploración y análisis, realice:
   - Crear pruebas unitarias y de integración del sistema (incluyendo testeo web de respuestas http con REST APIs).
   - Almacenar todas las pruebas en el archivo `tests.py`.
   - Ejecutar las pruebas utilizando Pytest e integrarlo con Django.
4. **Aplicación de Dobles de Prueba**:
   - Implementar dobles de prueba donde sea necesario.
5. **Gestión de Archivos**:
   - Utilizar `tmp_path` y `tmp_path_factory` para la gestión de archivos temporales en las pruebas.
6. **Uso de Fixtures**:
   - Aplicar fixtures para la configuración y limpieza de pruebas (`setUp` y `tearDown`).
7. **Informe Escrito**:

   - Elaborar un informe escrito que contenga:
     - Introducción
     - Resumen ejecutivo con el resultado de todas las pruebas.
     - Sección detallada por cada prueba con:
       - Precondiciones
       - Procesos
       - Post-condiciones
       - Resultado (pasó/no pasó)
       - Recomendación de solución en caso de no pasar.
       - Solución implementada.

8. **Informe Digital con Allure**:
   - Utilizar Allure para generar un informe digital.
   - Incluir metadatos en cada prueba utilizando decoradores de Allure como `@allure.title`, `@allure.issue`, y todas las demás vistas en clase (consulte la página web de allure para más detalles).

# Ejecutar todas las pruebas con Allure

pytest --alluredir=./allure-results

# Generar reporte de Allure

allure generate allure-results --clean -o allure-report
