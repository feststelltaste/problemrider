---
title: Große Schätzungen für kleine Änderungen
description: Das Team liefert durchgängig große Zeitschätzungen für scheinbar kleine
  Änderungen, was auf zugrunde liegende Codekomplexität und Risiko hindeutet.
category:
- Code
- Process
related_problems:
- slug: high-technical-debt
  similarity: 0.7
- slug: difficult-developer-onboarding
  similarity: 0.7
- slug: fear-of-change
  similarity: 0.7
- slug: frequent-changes-to-requirements
  similarity: 0.7
- slug: high-bug-introduction-rate
  similarity: 0.7
- slug: slow-feature-development
  similarity: 0.7
solutions:
- architecture-roadmap
- regression-testing
- capacity-based-planning
- mikado-method
- small-change-batches
- change-impact-analysis
- definition-of-ready
- preparatory-refactoring
- debt-remediation-estimation
- technical-debt-assessment
- automated-code-migration
- duplication-detection
layout: problem
lang: de
en_slug: large-estimates-for-small-changes
---

## Description
Wenn kleine, scheinbar einfache Änderungen durchgängig als langwierig in der Implementierung geschätzt werden, ist dies ein starker Indikator für zugrunde liegende Probleme in der Codebasis. Dieses Phänomen, oft als "hohe Änderungskosten" bezeichnet, deutet darauf hin, dass das System starr und brüchig geworden ist. Das Entwicklungsteam navigiert wahrscheinlich durch ein Minenfeld technischer Schulden, in dem jede Modifikation das Risiko unvorhergesehener Nebeneffekte birgt. Dieses Problem kann die Fähigkeit eines Teams lähmen, auf sich ändernde Geschäftsbedürfnisse zu reagieren, und kann eine wesentliche Quelle der Frustration sowohl für Entwickler als auch für Stakeholder sein.

## Indicators ⟡
- Ein einfacher Bugfix wird auf Tage oder Wochen geschätzt.
- Stakeholder sind überrascht von den hohen Kosten kleinerer Feature-Anfragen.
- Das Team verbringt mehr Zeit in Meetings mit der Diskussion der Risiken einer Änderung als mit deren tatsächlicher Umsetzung.
- Es gibt eine merkliche Zurückhaltung des Teams, Aufgaben zu übernehmen, die die Modifikation bestehenden Codes beinhalten.

## Symptoms ▲

- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Geschäfts-Stakeholder werden frustriert, wenn scheinbar einfache Änderungen unverhältnismäßig viel Zeit und Kosten erfordern, was das Vertrauen in das Entwicklungsteam untergräbt.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Wenn jede kleine Änderung erheblichen Aufwand erfordert, sinkt das Gesamttempo der Feature-Lieferung dramatisch.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Der hohe Aufwand, der selbst für kleinere Modifikationen erforderlich ist, treibt die Gesamtkosten der Entwicklungsarbeit direkt in die Höhe.
- [Vertrauensverlust bei Stakeholdern](vertrauensverlust-bei-stakeholdern.md)
<br/>  Wiederholt große Schätzungen für kleine Arbeit untergraben das Geschäftsvertrauen in die Fähigkeit des Teams, effizient zu liefern.
- [Probleme mit der Glaubwürdigkeit der Planung](probleme-mit-der-glaubwuerdigkeit-der-planung.md)
<br/>  Wenn Schätzungen unverhältnismäßig zur scheinbaren Arbeit erscheinen, hinterfragen Stakeholder die Zuverlässigkeit aller zukünftigen Planungen und Schätzungen.

## Causes ▼

- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine brüchige Codebasis, bei der Änderungen riskieren, andere Teile zu brechen, zwingt Teams, Schätzungen für umfangreiches Testen und Risikominderung aufzublähen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Komponenten bedeuten, dass selbst kleine Änderungen sich über viele Teile des Systems ausbreiten, was legitim erheblichen Aufwand erfordert.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne Tests zur Verifikation von Änderungen müssen Entwickler manuell verifizieren, dass Modifikationen nichts brechen, was den geschätzten Aufwand erheblich erhöht.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte Abkürzungen und Designkompromisse erschweren die Arbeit mit der Codebasis, was den Aufwand selbst für kleinere Änderungen aufbläht.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Wenn Code schwer verständlich ist, brauchen Entwickler zusätzliche Zeit, um das System zu verstehen, bevor sie Änderungen vornehmen, was Schätzungen in die Höhe treibt.

## Detection Methods ○
- **Analyse von Schätzungstrends:** Nachverfolgung der Schätzungen für Aufgaben ähnlicher Komplexität über die Zeit. Ein durchgängiger Anstieg der Schätzungen ist ein Warnsignal.
- **Vergleich geschätzter vs. tatsächlicher Zeit:** Wenn die tatsächlich benötigte Zeit zur Fertigstellung von Aufgaben durchgängig viel höher ist als die Schätzungen, deutet dies darauf hin, dass das Team mit unvorhergesehener Komplexität kämpft.
- **Entwicklerfeedback:** Befragung von Entwicklern, warum ihre Schätzungen so hoch sind. Ihre Antworten deuten oft auf die Grundursachen hin.
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge zur Messung der Codekomplexität. Hohe Komplexitätswerte korrelieren oft mit hohen Änderungskosten.

## Examples
Ein Produktmanager fordert eine kleine Änderung an der Benutzeroberfläche an: das Hinzufügen eines neuen Felds zu einem Formular. Das Entwicklungsteam schätzt, dass dies zwei Wochen zur Implementierung braucht. Der Produktmanager ist schockiert, da er erwartete, dass es eine einfache Ein-Tages-Aufgabe ist. Die Entwickler erklären, dass das Formular an mehreren Stellen in der Anwendung genutzt wird und das zugrunde liegende Datenmodell eng mit anderen Teilen des Systems gekoppelt ist. Jede Änderung am Formular erfordert umfangreiches Testen, um sicherzustellen, dass nichts anderes bricht. Dies ist ein klassisches Beispiel dafür, wie eine brüchige Codebasis zu großen Schätzungen für kleine Änderungen führen kann.
