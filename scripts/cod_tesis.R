# ============================================================
# 📊 GENERADOR DE DATASET DE TITULARES ESPAÑOLES (2014–2024)
# Versión para tesis: scraping + medio real + AFINN ponderado
# ============================================================

# ------------------------------------------------------------
# 🔧 0. CONFIGURACIÓN INICIAL
# ------------------------------------------------------------

setwd("C:/Users/JORGEJOSÉZAMORA/OneDrive - A2 Capital Iberia SL/Escritorio/Jorge/Scraping/Ejecución_R")
cat("📂 Carpeta de trabajo:", getwd(), "\n")

# ------------------------------------------------------------
# 1️⃣ LIBRERÍAS
# ------------------------------------------------------------

library(tidyRSS)
library(dplyr)
library(stringr)
library(tidytext)
library(openxlsx)

# ------------------------------------------------------------
# 2️⃣ DEFINICIÓN DE CATEGORÍAS, SUJETOS Y EVENTOS
# ------------------------------------------------------------

sujetos <- list(
  politica = c("Pedro Sánchez","Mariano Rajoy","Pablo Iglesias","Santiago Abascal","Yolanda Díaz",
               "Isabel Díaz Ayuso","Feijóo","PSOE","PP","VOX","Podemos","Gobierno","Congreso",
               "Senado","Cataluña","independentismo","Moncloa","elecciones generales"),
  
  sociedad = c("jóvenes","mujeres","inmigrantes","menores","policía","profesores","sanitarios",
               "trabajadores","familias","estudiantes","pensionistas","sanidad pública","educación",
               "violencia de género","paro","manifestaciones"),
  
  economia = c("Amazon","Mercadona","Inditex","Repsol","Banco Santander","BBVA","Telefónica",
               "Iberdrola","CNMC","inflación","PIB","consumo digital","coste de vida",
               "hogares españoles","crisis económica"),
  
  cultura = c("Netflix","YouTube","TikTok","Twitter","Facebook","Instagram","Rosalía","Shakira",
              "Ibai Llanos","El Rubius","Belén Esteban","Elon Musk","Greta Thunberg","Javier Milei"),
  
  medioambiente = c("COVID-19","OMS","Ministerio de Sanidad","científicos","vacuna","Pfizer",
                    "clima","calor","sequía","contaminación","cambio climático","incendios",
                    "inundaciones","pandemia"),
  
  deporte = c("Real Madrid","FC Barcelona","Atlético de Madrid","Vinicius","Mbappé","Messi",
              "Cristiano Ronaldo","Rubiales","Luis Enrique","Carlo Ancelotti","Xavi Hernández",
              "Selección Española","Liga de Campeones","UEFA","FIFA","Juegos Olímpicos",
              "Tenis","Rafa Nadal","Carlos Alcaraz","Fernando Alonso")
)

eventos_relacionados <- list(
  politica = c("corrupción","escándalo","dimisión","fraude","mentira","odio","protestas",
               "investigación","polémica","juicio","crisis política"),
  
  sociedad = c("crisis","manifestaciones","protestas","violencia","discriminación","suicidio",
               "asesinato","acoso","miedo","tensión social","odio","xenofobia"),
  
  economia = c("paro","recesión","subida de precios","beneficios","despidos","inflación",
               "fraude fiscal","colapso","deuda","estafa","recuperación económica"),
  
  cultura = c("escándalo","acusación","cancelación","odio","censura","fake news","viral",
              "trending topic","controversia","insulto","acoso online","boicot"),
  
  medioambiente = c("emergencia","alerta","muerte","crisis sanitaria","colapso","catástrofe",
                    "protestas","tragedia","desastre","fallecidos","alarma","rebrote","confinamiento"),
  
  deporte = c("victoria","derrota","escándalo","sanción","acusación","polémica","lesión","violencia",
              "protestas","tragedia","investigación","corrupción","dimisión","crisis",
              "acusaciones de racismo")
)

# ------------------------------------------------------------
# 3️⃣ GENERACIÓN DE COMBINACIONES (2014–2024)
# ------------------------------------------------------------

meses <- c("enero","febrero","marzo","abril","mayo","junio",
           "julio","agosto","septiembre","octubre","noviembre","diciembre")
anios <- 2014:2024

combinaciones <- bind_rows(lapply(names(sujetos), function(categoria) {
  expand.grid(
    sujeto   = sujetos[[categoria]],
    evento   = eventos_relacionados[[categoria]],
    categoria = categoria,
    stringsAsFactors = FALSE
  )
}))

combinaciones_finales <- merge(
  combinaciones,
  expand.grid(mes = meses, anio = anios),
  all = TRUE
)

combinaciones_finales$busqueda <- paste(
  combinaciones_finales$sujeto,
  combinaciones_finales$evento,
  combinaciones_finales$mes,
  combinaciones_finales$anio
)

cat("🔍 Total combinaciones generadas:", nrow(combinaciones_finales), "\n")

# ------------------------------------------------------------
# 4️⃣ FUNCIÓN DE SCRAPING (⚠️ TAREA LARGA)
# ------------------------------------------------------------

crear_url <- function(tema) {
  base <- "https://news.google.com/rss/search?q="
  query <- gsub(" ", "+", tema)
  paste0(base, query, "&hl=es&gl=ES&ceid=ES:es")
}

scrapear_categoria_anio <- function(nombre_categoria, anio_filtro) {
  cat("\n⏳ Procesando categoría:", nombre_categoria, "| Año:", anio_filtro, "\n")
  
  combinaciones_filtradas <- combinaciones_finales %>%
    filter(categoria == nombre_categoria, anio == anio_filtro)
  
  temas <- combinaciones_filtradas$busqueda
  
  lista_noticias <- lapply(seq_along(temas), function(i) {
    tema <- temas[i]
    cat("  → [", i, "/", length(temas), "]", tema, "\n")
    Sys.sleep(runif(1, 1, 2))  # Pausa para evitar bloqueo
    
    url  <- crear_url(tema)
    feed <- tryCatch(tidyfeed(url), error = function(e) data.frame())
    if (nrow(feed) == 0) return(NULL)
    
    colnames(feed) <- tolower(colnames(feed))
    if ("entry_title"     %in% colnames(feed)) feed$item_title    <- feed$entry_title
    if ("entry_link"      %in% colnames(feed)) feed$item_link     <- feed$entry_link
    if ("entry_published" %in% colnames(feed)) feed$item_pub_date <- feed$entry_published
    
    if (!all(c("item_title","item_link","item_pub_date") %in% colnames(feed))) return(NULL)
    
    feed %>%
      select(item_title, item_link, item_pub_date) %>%
      mutate(tema = tema)
  })
  
  lista_noticias <- Filter(Negate(is.null), lista_noticias)
  if (length(lista_noticias) == 0) return(NULL)
  
  noticias <- bind_rows(lista_noticias) %>%
    distinct(item_title, .keep_all = TRUE) %>%
    mutate(
      categoria = nombre_categoria,
      anio      = anio_filtro
    )
  
  saveRDS(noticias, paste0("noticias_", nombre_categoria, "_", anio_filtro, ".rds"))
  cat("✅ Guardado: noticias_", nombre_categoria, "_", anio_filtro, "\n")
  return(noticias)
}

# ------------------------------------------------------------
# 5️⃣ BUCLE DE SCRAPING (PUEDES COMENTARLO SI YA TIENES LOS .RDS)
# ------------------------------------------------------------

# categorias <- names(sujetos)
# for (cat in categorias) {
#   for (a in anios) {
#     tryCatch({
#       scrapear_categoria_anio(cat, a)
#     }, error = function(e) {
#       cat("⚠️ Error en:", cat, "-", a, "|", conditionMessage(e), "\n")
#     })
#   }
# }

# ------------------------------------------------------------
# 6️⃣ EXTRAER FUENTE REAL — SEGUIMIENTO DE REDIRECCIÓN
# ------------------------------------------------------------

library(httr)

resolver_fuente <- function(link_google){
  # si el link es NA
  if (is.na(link_google)) return(NA)
  
  # intentar seguir redirección
  resp <- tryCatch(
    httr::GET(link_google, timeout(5)), 
    error = function(e) return(NA)
  )
  
  # url final después de redirección
  final_url <- tryCatch(resp$url, error = function(e) NA)
  
  return(final_url)
}

cat("\n🔎 Resolviendo fuentes reales (esto puede tardar un poco)...\n")

noticias_total$fuente_url <- sapply(noticias_total$item_link, resolver_fuente)

# Extraer dominio real del medio
noticias_total <- noticias_total %>%
  mutate(
    dominio_real = str_extract(fuente_url, "(?<=//)[^/]+"),
    dominio_real = tolower(str_replace(dominio_real, "^www\\.", "")),
    medio = dominio_real
  )

cat("\n📌 Ejemplo de FUENTES detectadas:\n")
print(head(noticias_total[, c("item_title", "medio")], 10))



# ------------------------------------------------------------
# 7️⃣ TOKENIZACIÓN + AFINN-ES (PONDERADO)
# ------------------------------------------------------------

afinn_es <- read.csv("lexico_afinn.csv", stringsAsFactors = FALSE, encoding = "UTF-8")

# Asumimos columnas: palabra, puntuacion, word
names(afinn_es) <- c("palabra", "puntuacion", "word")

afinn_es <- afinn_es %>%
  mutate(
    palabra = tolower(palabra),
    puntuacion = as.numeric(puntuacion)
  ) %>%
  distinct(palabra, .keep_all = TRUE)

titulares <- noticias_total %>%
  filter(!is.na(item_title) & nchar(item_title) > 3) %>%
  mutate(item_title = str_replace_all(item_title, "[^[:alnum:]\\s]", " ")) %>%
  unnest_tokens(palabra, item_title)

sentimiento_afinn <- titulares %>%
  inner_join(afinn_es, by = "palabra") %>%
  group_by(id) %>%
  summarise(
    palabras_encontradas = n(),
    suma_sentimientos    = sum(puntuacion, na.rm = TRUE),
    sentimiento_ponderado = suma_sentimientos / palabras_encontradas,
    .groups = "drop"
  )


# ------------------------------------------------------------
# 8️⃣ EXPORTACIÓN FINAL (CON VISTA PREVIA) EXCEL
# ------------------------------------------------------------

noticias_afinn <- noticias_total %>%
  left_join(sentimiento_afinn, by = "id") %>%
  mutate(
    sentimiento_ponderado = ifelse(is.na(sentimiento_ponderado), 0, sentimiento_ponderado),
    item_pub_date = as.Date(item_pub_date),
    anio_pub = format(item_pub_date, "%Y"),
    mes_pub  = format(item_pub_date, "%m")
  )

cat("\n🔎 PREVISUALIZACIÓN DE LAS PRIMERAS FILAS DEL DATASET FINAL:\n")
print(head(noticias_afinn))

fecha_actual <- format(Sys.Date(), "%Y%m%d")

write.xlsx(
  noticias_afinn,
  paste0("noticias_completas_", fecha_actual, "_PONDERADO.xlsx")
)

cat("\n📁 EXPORTADO:", nrow(noticias_afinn), "titulares.\n")
cat("🎉 Dataset final generado con éxito.\n")


# ============================================================
#  EXPORTACIÓN A CSV COMPLETO (SIN LÍMITE DE FILAS)
# ============================================================

cat("📁 Exportando CSV completo... Esto puede tardar unos segundos.\n")

nombre_csv <- paste0("noticias_completas_", format(Sys.Date(), "%Y%m%d"), ".csv")

write.csv(
  noticias_afinn,
  file = nombre_csv,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("✔ CSV completo generado:", nombre_csv, "\n")
cat("✔ Total de filas exportadas:", nrow(noticias_afinn), "\n")

# ============================================================
#  EXPORTACIÓN A TXT (TSV — tab delimited)
# ============================================================

nombre_tsv <- paste0("noticias_completas_", format(Sys.Date(), "%Y%m%d"), ".txt")

cat("📄 Exportando TXT (tabulaciones)...\n")

write.table(
  noticias_afinn,
  file = nombre_tsv,
  sep = "\t",
  row.names = FALSE,
  quote = TRUE,
  fileEncoding = "UTF-8"
)

cat("✔ TXT generado:", nombre_tsv, "\n")


# ============================================================
#  EXPORTACIÓN A TXT COMPRIMIDO (GZIP)
# ============================================================

nombre_txt_gz <- paste0("noticias_completas_", format(Sys.Date(), "%Y%m%d"), ".txt.gz")

cat("📦 Exportando TXT comprimido...\n")

gz_con <- gzfile(nombre_txt_gz, "w")

write.table(
  noticias_afinn,
  gz_con,
  sep = "\t",
  row.names = FALSE,
  quote = TRUE,
  fileEncoding = "UTF-8"
)

close(gz_con)

cat("✔ TXT comprimido generado:", nombre_txt_gz, "\n")

# ============================================================
# END
# ============================================================