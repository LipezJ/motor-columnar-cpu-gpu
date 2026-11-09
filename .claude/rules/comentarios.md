---
paths:
  - "**/*.cpp"
  - "**/*.hpp"
  - "**/*.cu"
  - "**/*.cuh"
  - "**/*.py"
  - "**/*.sh"
  - "**/Makefile"
---

# Comentarios

Por defecto no se comenta. El comentario es la excepción y quien lo escribe carga con justificarlo.

La justificación es una sola: **el lector no puede llegar a ese hecho desde el código.** En este proyecto no hay ADRs ni documentación de diseño aparte del `README.md`, así que el porqué de cada decisión de diseño vive en el código, junto a lo que decide. El `README.md` es dueño solo de la pregunta del proyecto, la lista de backends y consultas, y los entornos; eso no se repite en el código.

Que el comentario sea verdad no es mérito. Uno que explica lo que el código ya dice se borra aunque sea correcto: no se acorta, no se reescribe mejor, no se deja porque no molesta. Ante la duda, se borra.

## Qué se queda

La lista es cerrada. Un comentario que no sea una de estas cosas se borra.

1. **Una restricción que no se ve desde el código.** Que dos valores tengan que cambiar juntos (un código de diccionario en `scripts/gen_datos.py` y su constante en C++), que una suma desborde por encima de cierto factor de escala, que un kernel asuma un tamaño de bloque, que una condición de carrera sea benigna y por qué.
2. **Qué es una cosa, cuando su nombre no lo dice.** Que un arreglo sea la tabla hash del lado *build*, que una columna guarde centavos y no unidades, que una fecha sea días desde 1970-01-01.
3. **Por qué esto y no lo evidente.** La alternativa descartada y su motivo. Aquí es el comentario más importante, porque no hay ADR que lo guarde: por qué direccionamiento abierto y no `std::unordered_map`, por qué el tamaño de *morsel* es el que es, por qué la agregación de Q1 en GPU va primero a memoria compartida. Si el motivo es una medición, se dice qué se midió.
4. **De dónde sale una técnica**, cuando viene de un paper: autor, venue y año (`Leis et al., SIGMOD 2014`). Es identidad del algoritmo, no comportamiento, y es lo que el jurado pregunta.
5. **Qué significa una opción de configuración o de línea de comandos**: unidades, valores admitidos y qué se rompe al cambiarla.
6. **Un `TODO` con dueño y contexto.**

Se escriben como especificación:

- **Cada cosa por su identificador.** `TAM_MORSEL`, nunca "el tamaño de arriba".
- **Una afirmación por frase.** Dos hechos son dos frases.
- **Un motivo por frase como mucho, y entre paréntesis.**
- **Junto a lo que explica.** El motivo de una rama va sobre su condición. El comentario de cabecera de una función dice lo que obtiene quien la llama, no cómo lo hace.

Falta un comentario cuando el código depende de algo de esta lista y no lo dice. Ese hueco es defecto igual que un comentario de más.

## Qué sobra

Basta que una pregunta falle para borrarlo.

1. **¿El hecho ya está en el código que comenta?** En el nombre, en la firma, en la línea siguiente.
2. **¿El hecho tiene dueño en otro archivo?** El `README.md`, el comentario de cabecera de la función que se invoca, el mensaje del commit. Se escribe una sola vez, en su casa.
3. **¿Se vuelve falso con un refactor que no cambie el comportamiento?** Reordenar dos llamadas, renombrar una variable local, partir una función. Si lo vuelve falso, era implementación.

| Forma | Falla |
|---|---|
| Narra el arreglo recién hecho | 2 (va en el commit) |
| Enumera los pasos de la función | 3 |
| Anuncia lo que viene ("aquí se filtra") | 1 |
| Traduce al español la línea siguiente | 1 |
| Repite en prosa el nombre que tiene al lado | 1 |
| Separadores decorativos (`// ======`) | no afirma nada; se quita |

Un bloque de código comentado no se deja: si algo se quita, se quita.

## El comentario que se toca

Editar un comentario lo adopta entero: el bloque completo vuelve a pasar por #Qué sobra. Un comentario que un cambio vuelve falso se corrige o se borra en el mismo commit.

## Justificar una omisión

Un comentario que descarta algo por su costo ("habría que duplicar", "no compensa") exige comprobar antes que la pieza no exista ya en el repo. Si es falso, sostiene una implementación peor con apariencia de decisión deliberada.

## Forma

Tildes y ñ correctas en comentarios y en los mensajes que un programa imprime. Los identificadores van en ASCII, en español cuando son nuestros (`tam_morsel`, `tabla_hash`) y en inglés solo cuando son nombres establecidos (`morsel`, `build`, `probe`, `tile`).
