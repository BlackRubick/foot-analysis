# Trazabilidad de Especificacion - Cadenas Musculares

- Items totales: 32
- Items con cobertura en reglas actuales: 28
- Items pendientes declarados: 4
- Cumplimiento global aproximado: 87.5%

| Area | Rasgo | Codigos Regla | Cubierto | Nota |
|---|---|---|---|---|
| Flexion | Genu flexum | genu_flexum | SI | Regla directa por angulo de rodilla |
| Flexion | Sacro vertical | sacro_vertical | SI | Basado en flecha sacra |
| Flexion | Cifosis | cifosis, hipercifosis | SI | Basado en D7 y curva toracica |
| Flexion | Esternon hundido / pectus | pectus_excavatum | SI | Aproximado por angulo de Charpy |
| Flexion | Inversion cervical | inversion_cervical | SI | Medicion angular cervical |
| Flexion | Retroversion pelvica | retroversion_pelvica | SI | Proxy pelvis/sacro |
| Flexion | Coxis hacia adentro | coxis_hacia_adentro | SI | Inferido por flecha sacra |
| Flexion | Cierre de costillas | cierre_costillas, cierre_costal_global | SI | Charpy y patron de cierre |
| Flexion | Proyeccion anterior cabeza | proyeccion_anterior_cabeza | SI | Angulo craneovertebral |
| Flexion | Rectificacion lumbar | rectificacion_lumbar, rectificacion_lumbar_baja | SI | Flecha lumbar |
| Flexion | Valgo de rodilla | valgo_rodilla | SI | Aproximacion por angulo frontal |
| Flexion | Rotacion interna cadera | rotacion_interna_cadera | SI | Proxy por pie-talón |
| Flexion | Aduccion/rotacion interna brazos | aduccion_brazos, rotacion_interna_brazos | SI | Distancia a linea media + mano |
| Flexion | Descenso de hombros | descenso_hombros | SI | Aproximacion vertical hombro-cabeza |
| Flexion | Flexion MMII | flexion_msls, ankle_posteriorizado_flexion, knee_posteriorizada_flexion | SI | Integracion rodilla y Barré |
| Flexion | Cierre mandibular | cierre_mandibula, mandibula_cerrada_global | SI | Angulo mandibular |
| Extension | Genu recurvatum | genu_recurvatum | SI | Regla directa por angulo de rodilla |
| Extension | Sacro horizontal | sacro_horizontal | SI | Flecha sacra elevada |
| Extension | Dorso plano | dorso_plano, espalda_plana | SI | Flecha dorsal reducida |
| Extension | Rectificacion cervical | rectificacion_cervical | SI | Medicion angular cervical |
| Extension | Bascula posterior de cabeza | bascula_posterior_cabeza | SI | Angulo craneovertebral alto |
| Extension | Esternon horizontal | esternon_horizontal | SI | Proxy por apertura clavicular |
| Extension | Apertura mandibular | apertura_mandibula, mandibula_abierta_global | SI | Angulo mandibular |
| Extension | Anteversion pelvica | anteversion_pelvica | SI | Proxy pelvis/sacro |
| Extension | Hiperlordosis | hiperlordosis, hiperlordosis_baja | SI | Flecha lumbar alta |
| Extension | Extension MMII | extension_msls, ankle_antepulsion_extension, knee_antepulsion_extension | SI | Integracion rodilla y Barré |
| Extension | Ascenso hombros | ascenso_hombros | SI | Aproximacion vertical hombro-cabeza |
| Extension | Rotacion externa | rotacion_externa, rotacion_externa_brazos, varo_rodilla | SI | Versiones opuestas |
| Pendiente | Hallux valgus | - | NO | Requiere pipeline dedicado de pie frontal |
| Pendiente | Pie cavo/supino | - | NO | Requiere integrar huella plantar al modulo de cadenas |
| Pendiente | Dedos en garra | - | NO | Requiere segmentacion detallada de dedos en huella |
| Pendiente | Validacion clinica de umbrales | - | NO | Se necesita dataset etiquetado por especialista |
