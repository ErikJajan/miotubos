# Segmentación de miotubos

El notebook [eda_miotubos.ipynb](eda_miotubos.ipynb) contiene el análisis
exploratorio, la auditoría de fuga y los resultados de la línea base clásica.
El [reporte](output/pdf/reporte_miotubos.pdf) resume los hallazgos.

## Reproducir el notebook

1. Coloca los datos con la estructura `vt/vt/{images,masks_png,labels,coco,verification}`.
2. Con Python 3.10 o superior, crea un entorno.

   ```sh
   python -m venv .venv
   ```

   Activa el entorno con `.venv\Scripts\Activate.ps1` en PowerShell o
   `source .venv/bin/activate` en bash. Instala las dependencias con
   `python -m pip install -r requirements.txt`.
3. Abre el notebook desde la raíz del repositorio, selecciona ese entorno como
   kernel de Python y ejecuta todas las celdas. En JupyterLab, instala también
   `jupyterlab` en el entorno si necesitas la interfaz.

El directorio `resultados/` contiene manifiestos, auditoría, tabla de desbalance,
métricas y figuras. Las rutas de los manifiestos son relativas al directorio
de datos. Los archivos originales y sus máscaras permanecen en `vt/`.

## Regenerar las particiones desde un CSV

En la copia disponible del reto solo hay anotaciones COCO y YOLO, junto con
las máscaras rasterizadas. El notebook exporta el COCO original a
`resultados/catalogo_coco.csv`. Ese catálogo se incluye para poder reproducir
la partición desde el CSV efectivamente utilizado.

```sh
python pipeline.py --root vt/vt --csv resultados/catalogo_coco.csv --out resultados
```

El pipeline acepta otro CSV con las columnas `image_id,image,mask,width,height,instances,group_id`.
Usa en `group_id` el identificador experimental compartido cuando esté disponible.
La semilla por defecto es 2026 y la fracción objetivo de validación es 0.20.
Si se omite `--csv`, el pipeline deriva el catálogo del COCO de origen.

El reparto mantiene juntas las imágenes de cada grupo y las copias detectadas.
Los UUID de los nombres son una aproximación de serie. Cuando faltan, el
catálogo los marca como `origen-no-confirmado` y el pipeline mantiene ese grupo
en entrenamiento. `split_audit.json` documenta los cruces detectados. El PDF
del reto propone ampliar el estudio con un modelo profundo y una prueba
independiente. El notebook cubre el EDA, el pipeline y la línea base solicitados.
