---
title: Langlebige Feature-Branches
description: Code wird nicht zeitnah reviewt und gemerged, was zu Integrationsproblemen
  und erhöhtem Risiko führt.
category:
- Code
- Process
related_problems:
- slug: large-feature-scope
  similarity: 0.75
- slug: merge-conflicts
  similarity: 0.7
- slug: slow-feature-development
  similarity: 0.7
- slug: large-pull-requests
  similarity: 0.65
- slug: high-technical-debt
  similarity: 0.6
- slug: difficult-developer-onboarding
  similarity: 0.6
solutions:
- feature-flags
- continuous-integration
- trunk-based-development
- small-change-batches
- code-review-guidelines
- work-in-progress-limits
- preparatory-refactoring
- ci-cd-pipeline
- mikado-method
layout: problem
lang: de
en_slug: long-lived-feature-branches
---

## Description
Langlebige Feature-Branches sind ein häufiges Problem in Teams, die ein Branching-Modell für die Entwicklung nutzen. Wenn ein Feature-Branch über einen längeren Zeitraum vom Hauptbranch getrennt gehalten wird, kann das Zurückmergen schwierig und riskant werden. Je länger ein Branch lebt, desto mehr weicht er vom Hauptbranch ab, was die Wahrscheinlichkeit von Merge-Konflikten erhöht und die Integration der Änderungen erschwert. Dies kann zu einem „Merge-Hölle"-Szenario führen, bei dem viel Zeit mit der Auflösung von Konflikten verbracht wird, statt Wert zu liefern.

## Indicators ⟡
- Feature-Branches sind oft Tage oder Wochen alt.
- Das Mergen eines Feature-Branches ist ein großes Ereignis, das viel Koordination erfordert.
- Das Team hat ständig mit Merge-Konflikten zu kämpfen.
- Das Team hat Angst, Feature-Branches zu mergen, aus Furcht, etwas zu brechen.

## Symptoms ▲

- [Merge-Konflikte](merge-konflikte.md)
<br/>  Je länger ein Branch vom Hauptstrang abweicht, desto wahrscheinlicher häufen sich widersprüchliche Änderungen an, was schmerzhafte Merge-Konflikte erzeugt.
- [Integrationsschwierigkeiten](integrationsschwierigkeiten.md)
<br/>  Code, der über längere Zeit isoliert entwickelt wurde, wird strukturell inkompatibel mit Änderungen am Hauptstrang, was die Integration kostspielig macht.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Große Merges von langlebigen Branches führen viele Änderungen auf einmal ein, was die Wahrscheinlichkeit subtiler Regressionen erhöht.
- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Wenn parallele Entwicklung am Hauptstrang den Ansatz eines Branches inkompatibel macht, ist erhebliche Nacharbeit vor dem Mergen nötig.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Langlebige Branches häufen viele Änderungen an, was zu großen Pull Requests führt, die schwer effektiv zu reviewen sind.

## Causes ▼

- [Großer Feature-Umfang](grosser-feature-umfang.md)
<br/>  Features mit übermäßig breitem Umfang brauchen länger zur Implementierung, was natürlicherweise die Lebensdauer von Branches verlängert.
- [Lange Build- und Testzeiten](lange-build-und-testzeiten.md)
<br/>  Langsame CI-Pipelines entmutigen häufige Integration, da Entwickler die langen Feedback-Zyklen des häufigen Mergens vermeiden.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Engpässe im Code-Review-Prozess verzögern Merges und zwingen Branches, länger zu leben als beabsichtigt.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Teams verzögern das Mergen, weil sie fürchten, Breaking Changes einzuführen, und halten Branches länger getrennt.

## Detection Methods ○

- **Analyse des Versionskontrollsystems:** Überwachung des Alters und der Größe von Feature-Branches in Ihrem Git-Repository.
- **Code-Review-Metriken:** Nachverfolgung der Zeit, die Pull Requests bis zum Review und Merge benötigen.
- **Build-/Deployment-Häufigkeit:** Beobachtung, wie oft der Hauptbranch gebaut und deployt wird.
- **Entwickler-Feedback:** Befragung von Entwicklern zu ihren Erfahrungen mit Merge-Konflikten und Integrationsherausforderungen.

## Examples
Ein Team entwickelt ein bedeutendes neues Modul für eine Anwendung. Die Entwicklung dauert drei Monate auf einem einzigen Feature-Branch. Als es Zeit zum Mergen ist, gibt es Hunderte Konflikte mit dem Hauptbranch, und das Team verbringt Wochen damit, sie aufzulösen, was das Release verzögert. In einem anderen Fall arbeitet ein Entwickler mehrere Wochen an einem neuen Feature, ohne seine Änderungen zu pushen oder einen Pull Request zu erstellen. Währenddessen macht ein anderer Entwickler eine verwandte Änderung am Hauptbranch. Als der erste Entwickler schließlich versucht zu mergen, sind seine Änderungen inkompatibel, was erhebliche Nacharbeit erfordert. Dieses Problem ist oft ein Symptom eines Teams, das kontinuierliche Integration oder agile Entwicklungspraktiken noch nicht vollständig übernommen hat. Es kann zu erheblichen technischen Schulden führen und den gesamten Entwicklungsprozess verlangsamen.
