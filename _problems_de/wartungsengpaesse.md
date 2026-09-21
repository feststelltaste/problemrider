---
title: Wartungsengpässe
description: Eine Situation, in der nur eine kleine Anzahl von Entwicklern Änderungen
  an einem kritischen Teil des Systems vornehmen kann.
category:
- Code
- Process
- Team
related_problems:
- slug: bottleneck-formation
  similarity: 0.75
- slug: review-bottlenecks
  similarity: 0.7
- slug: single-points-of-failure
  similarity: 0.65
- slug: maintenance-paralysis
  similarity: 0.65
- slug: work-queue-buildup
  similarity: 0.65
- slug: work-blocking
  similarity: 0.65
solutions:
- architecture-roadmap
- clear-ownership-model
- contract-testing
- code-hotspot-analysis
- work-in-progress-limits
- knowledge-rotation
- team-boundaries-aligned-to-architecture
- improvement-budget
- code-reading-sessions
- technical-debt-assessment
layout: problem
lang: de
en_slug: maintenance-bottlenecks
---

## Description
Ein Wartungsengpass tritt auf, wenn eine kleine Anzahl von Entwicklern oder sogar ein einzelner Entwickler die einzigen sind, die das Wissen und die Expertise haben, um einen kritischen Teil des Systems zu warten. Dies schafft einen Single Point of Failure und kann das Entwicklungstempo erheblich verlangsamen. Es setzt auch die Entwickler, die die Engpässe sind, unter erheblichen Stress.

## Indicators ⟡
- Eine kleine Anzahl von Entwicklern wird konsequent für die Arbeit an einem bestimmten Teil des Systems eingesetzt.
- Andere Entwickler zögern, Änderungen an diesem Teil des Systems vorzunehmen.
- Die Entwickler, die die Engpässe sind, sind oft mit Arbeit überlastet.
- Es mangelt an Dokumentation für diesen Teil des Systems.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn nur wenige Personen kritische Systemteile modifizieren können, stauen sich Arbeiten auf, und die gesamte Entwicklungsgeschwindigkeit sinkt.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Die wenigen Entwickler, die Engpässe sind, werden mit Arbeit überlastet, was zu Stress und schließlich Burnout führt.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Wenn nur ein oder zwei Personen ein kritisches System warten können, schafft ihre Nichtverfügbarkeit einen direkten Single Point of Failure.
- [Verzögerte Problemlösung](verzoegerte-problemloesung.md)
<br/>  Fehlerbehebungen und Verbesserungen verzögern sich, weil sie auf die Verfügbarkeit der begrenzten Engpass-Entwickler warten müssen.
- [Arbeitsblockade](arbeitsblockade.md)
<br/>  Andere Teammitglieder sind daran gehindert, Fortschritte bei Aufgaben zu machen, die die Engpass-Systemkomponenten betreffen.

## Causes ▼

- [Wissenssilos](wissenssilos.md)
<br/>  Wissen, das bei wenigen Personen über kritische Systemteile konzentriert ist, schafft den Engpasszustand.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Wenn Systemwissen nur in den Köpfen von Entwicklern statt in Dokumentation existiert, können neue Entwickler nicht zu diesen Bereichen beitragen.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Wenn ein System veraltete Technologien nutzt, haben nur wenige Entwickler die erforderlichen Fähigkeiten, es zu warten.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Komplexer, schlecht dokumentierter Code entmutigt andere Entwickler davon, das System zu lernen und daran zu arbeiten.

## Detection Methods ○
- **Bus-Faktor-Analyse:** Identifikation der Schlüsselentwickler, die die Einzigen sind, die wissen, wie man an einem kritischen Teil des Systems arbeitet.
- **Code-Ownership-Analyse:** Nutzung von Werkzeugen zur Identifikation der Entwickler, die die meisten Änderungen an einem bestimmten Teil des Systems vorgenommen haben.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, dass es Teile des Systems gibt, die sie zu ändern fürchten.

## Examples
Ein Unternehmen hat ein Legacy-Abrechnungssystem, das von einem einzigen Entwickler geschrieben wurde, der das Unternehmen seitdem verlassen hat. Jetzt versteht nur noch ein anderer Entwickler im Team, wie das System funktioniert. Dieser Entwickler wird ständig von seiner anderen Arbeit weggezogen, um Fehler zu beheben und Änderungen am Abrechnungssystem vorzunehmen. Die anderen Entwickler im Team haben Angst, das Abrechnungssystem anzufassen, weil sie es nicht verstehen und Angst haben, es zu brechen. Infolgedessen ist das Abrechnungssystem zu einem erheblichen Engpass für das Unternehmen geworden.
