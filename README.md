# Trabalhar com o GitHub

**PASSO 1:** Antes de começares a trabalhar vê em que branch estás:

git branch
Se já estiveres no teu branch, avança para o passo 2. Se não estiveres, muda para o teu branch com:

git checkout nome-do-teu-branch

**PASSO 2:**

git pull origin nome-branch-comum

Agora estás pronto para trabalhar à vontade

**PASSO 3:** Quando acabares o trabalho

salvar o trabalho no próprio PC (control+S)

git add .

git commit -m "mensagem explicativa do commit"

git push origin nome-do-teu-branch

**PASSO 4:** Atualizar o branch comum

git checkout nome-branch-comum

git pull origin nome-branch-comum

git merge nome-do-teu-branch

git push origin nome-branch-comum




# Data-Mining-Project
In this project, we acted as consultants for AIAI. Our task was to analyze customer loyalty membership data
and corresponding flight activity collected over a three-year period. Using these insights, we
developed a **data-driven segmentation strategy**.
The segmentation was approached from multiple perspectives, such as:
- Value-based segmentation, grouping customers according to their economic contribution.
- Behavioral segmentation, analyzing purchasing habits and travel behaviors.
- Demographic segmentation, categorizing customers by age, occupation, or other attributes to reveal
different interaction patterns.

The ultimate objective was to integrate these perspectives into a final segmentation framework that supports
AIAI in crafting a comprehensive marketing strategy.

