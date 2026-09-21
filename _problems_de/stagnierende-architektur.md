---
title: Stagnierende Architektur
description: Die Systemarchitektur entwickelt sich nicht weiter, um sich ändernde
  Geschäftsbedürfnisse zu erfüllen, weil größere Refactorings konsequent vermieden werden.
category:
- Architecture
- Code
- Process
related_problems:
- slug: system-stagnation
  similarity: 0.85
- slug: architectural-mismatch
  similarity: 0.7
- slug: resistance-to-change
  similarity: 0.7
- slug: technical-architecture-limitations
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.65
- slug: schema-evolution-paralysis
  similarity: 0.65
solutions:
- architecture-decision-records
- strangler-fig-pattern
- abstraction
- architecture-conformity-analysis
- architecture-documentation
- architecture-governance
- architecture-review-board
- architecture-workshops
- emulation
- forward-compatibility
- hexagonal-architecture
- high-availability-architectures
- microservices
- microservices-architecture
- modulith
- platform-independence
- platform-independent-programming-languages
- security-architecture-analysis
- security-by-design
- fitness-functions
- threat-modeling
layout: problem
lang: de
en_slug: stagnant-architecture
---

## Description

Stagnierende Architektur tritt auf, wenn das fundamentale Design und die Struktur eines Systems unverändert bleiben, trotz sich entwickelnder Geschäftsanforderungen, technologischer Fortschritte und Lehren aus operativer Erfahrung. Dies geschieht, wenn Teams konsequent architektonische Verbesserungen aufgrund wahrgenommener Risiken, Zeitbeschränkungen oder Komplexität vermeiden. Das Ergebnis ist ein System, das zunehmend nicht mehr mit aktuellen Bedürfnissen übereinstimmt, was es schwierig macht, neue Features effizient zu implementieren oder mit modernen Technologien zu integrieren.

## Indicators ⟡

- Kernarchitekturmuster haben sich trotz neuer Anforderungen seit Jahren nicht geändert
- Neue Features fühlen sich „angeflanscht" an statt natürlich integriert
- Entwickler erwähnen häufig, dass „das System dafür nicht designt wurde"
- Die Integration mit neuen Technologien erfordert umfangreiche Adapterschichten
- Die Systemarchitektur stammt aus einer Zeit vor aktuellen Geschäftsmodellen oder Nutzungsmustern

## Symptoms ▲

- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Eine Architektur, die sich nicht weiterentwickelt hat, wird zunehmend fehlausgerichtet zu aktuellen Geschäftsanforderungen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Features, die nicht zur veralteten Architektur passen, erfordern umfangreiche Workarounds, was die Entwicklung dramatisch verlangsamt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn die Architektur neue Anforderungen nicht natürlich berücksichtigen kann, erstellen Entwickler Workarounds, die sich über die Zeit anhäufen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Anflanschen neuer Funktionalität an eine veraltete Architektur schafft wachsende technische Schulden.

## Causes ▼

- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Teams vermeiden architektonische Weiterentwicklung, weil sie die Risiken und Störungen größerer Refactorings fürchten.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das konsequente Aufschieben von Refactoring verhindert, dass sich die Architektur weiterentwickelt, um sich ändernde Bedürfnisse zu erfüllen.
- [Zeitdruck](zeitdruck.md)
<br/>  Konstanter Lieferdruck verhindert, dass Teams Zeit in architektonische Verbesserungen investieren.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Widerstand gegen Veränderung verursacht direkt, dass die Architektur stagniert, da Teams die benötigte Modernisierung und Refactoring vermeiden.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die anhaltende Priorisierung unmittelbarer Feature-Lieferung durch das Management lässt keinen Raum für die anhaltende Investition, die zur Weiterentwicklung oder Modernisierung der Architektur nötig ist.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Wenn Schlüsselentscheidungen zur Architektur dauerhaft aufgeschoben werden und sich zu einem voneinander abhängigen Rückstand verdichten, hat die Architektur keinen Weg zur Weiterentwicklung.

## Detection Methods ○

- **Architektur-Review-Sitzungen:** Regelmäßige Bewertung, wie gut die aktuelle Architektur Geschäftsbedürfnisse bedient
- **Technologie-Stack-Analyse:** Vergleich des aktuellen Stacks mit Industriestandards und modernen Alternativen
- **Verfolgung der Feature-Entwicklungszeit:** Überwachung, ob ähnliche Features zunehmend mehr Zeit brauchen
- **Integrationskomplexitätsmetriken:** Messung des Aufwands, der zur Integration mit neuen Systemen oder Services erforderlich ist
- **Entwickler-Feedback:** Befragung des Teams zu architektonischen Schmerzpunkten und Beschränkungen

## Examples

Eine vor 8 Jahren mit einer traditionellen Drei-Schichten-Architektur gebaute E-Commerce-Plattform hat Schwierigkeiten, moderne Features wie Echtzeit-Bestandsaktualisierungen, personalisierte Empfehlungen und Mobile-First-Nutzererfahrungen zu implementieren. Das monolithische Design macht es schwierig, einzelne Komponenten zu skalieren, Microservices für neue Funktionalität zu implementieren oder ereignisgesteuerte Muster zu übernehmen. Neue Features wie Social-Media-Integration erfordern umfangreiche Workarounds, weil die ursprüngliche Architektur annahm, dass alle Nutzerinteraktionen über die Weboberfläche stattfinden würden. Ein weiteres Beispiel betrifft eine Finanzdienstleistungsanwendung, bei der die ursprüngliche Client-Server-Architektur die Implementierung moderner Sicherheitsmuster, Echtzeit-Betrugserkennung und Cloud-nativer Deployment-Strategien verhindert, was das Team zwingt, zunehmend komplexe Lösungen auf die unflexible Grundlage zu schichten.
