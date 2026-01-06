#!/bin/bash

# ============================
# Git auto commit & push
# ============================

# Mensaje por defecto
MSG=${1:-"auto commit"}

# Comprobar si hay cambios
if [[ -z $(git status --porcelain) ]]; then
    echo "✔ No hay cambios para commitear"
    exit 0
fi

echo "➕ Añadiendo cambios..."
git add .

echo "📝 Commit con mensaje: $MSG"
git commit -m "$MSG"

echo "🚀 Push a origin..."
git push

echo "✅ Done"
