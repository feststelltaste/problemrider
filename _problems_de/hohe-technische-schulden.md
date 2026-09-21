---
title: Hohe technische Schulden
description: Anhäufung von Design- oder Implementierungsabkürzungen, die langfristig
  zu erhöhten Kosten und Aufwand führen.
category:
- Code
- Process
related_problems:
- slug: increased-technical-shortcuts
  similarity: 0.75
- slug: invisible-nature-of-technical-debt
  similarity: 0.75
- slug: time-pressure
  similarity: 0.7
- slug: accumulation-of-workarounds
  similarity: 0.7
- slug: maintenance-overhead
  similarity: 0.7
- slug: accumulated-decision-debt
  similarity: 0.7
solutions:
- technical-debt-backlog
- architecture-governance
- architecture-review-board
- code-metrics
- risk-analysis
- code-quality-gates
- deprecation-strategy
- fitness-functions
- functional-debt-management
- third-party-dependency-check
- code-hotspot-analysis
- improvement-budget
- total-cost-of-ownership-transparency
- workaround-registry
- technical-debt-assessment
- debt-classification
- debt-remediation-estimation
- debt-accrual-analysis
- quality-ratchet
- cost-of-delay
- baseline-measurement
- risk-quantification
- value-hierarchy
- automated-code-migration
- large-scale-refactoring
- continuous-dependency-updates
- duplication-detection
- attribute-usage-analysis
- explicit-extension-points
- typed-schema-extraction
- variant-consolidation
layout: problem
lang: de
en_slug: high-technical-debt
---

## Description
Hohe technische Schulden sind die implizierten Kosten der Nacharbeit, die dadurch entstehen, jetzt eine einfache (begrenzte) Lösung zu wählen, statt einen besseren Ansatz zu verwenden, der länger dauern würde. Diese Schulden häufen sich an, wenn Organisationen es versäumen, dedizierte Zeit, Ressourcen oder Budget für die Verbesserung bestehender Codequalität, die Behebung technischer Schulden oder die Modernisierung der Systemarchitektur bereitzustellen. Dies schafft einen Kreislauf, in dem sich technische Schulden schneller anhäufen, als sie behoben werden können, was das System letztlich zunehmend schwerer und teurer zu warten macht. Technische Schulden können ein erheblicher Produktivitätshemmschuh sein und es schwierig und riskant machen, neue Features hinzuzufügen oder Änderungen an der Codebasis vorzunehmen.

## Indicators ⟡
- Das Team behebt ständig Fehler statt neue Features zu bauen.
- Es dauert lange, neue Entwickler einzuarbeiten.
- Das Team zögert, Code zu refaktorieren.
- Es gibt viel duplizierten Code.
- Die gesamte Entwicklungszeit ist neuen Features oder Fehlerbehebungen zugeteilt.
- Refactoring-Arbeit wird nur getan, wenn es absolut notwendig ist, um andere Features fertigzustellen.
- Technische-Schulden-Posten werden identifiziert, aber nie in der Sprint-Planung priorisiert.
- Entwickler äußern Frustration darüber, keine Zeit zu haben, Code "aufzuräumen".

## Symptoms ▲

- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Der ständige Kampf mit schuldenbelastetem Code, um selbst einfache Änderungen vorzunehmen, ist demoralisierend und trägt zu chronischer Frustration und Burnout bei.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Angehäufte Abkürzungen und Codekomplexität machen jede Änderung teurer, was die Gesamtkosten der Systemwartung erhöht.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Technische Schulden verlangsamen die Feature-Entwicklung, während Entwickler komplexen, brüchigen Code navigieren und bestehende Probleme umgehen müssen.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Komplexer, schlecht strukturierter Code ist anfälliger für Fehler, da Änderungen unvorhersehbare Nebeneffekte haben.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Eine mit technischen Schulden belastete Codebasis ist für neue Entwickler schwerer zu verstehen, aufgrund von Inkonsistenzen, Workarounds und Komplexität.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Hohe technische Schulden machen Änderungen riskant, was Entwickler und Management dazu bringt, Modifikationen am System zu widerstehen.
- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Teams verbringen so viel Aufwand mit der Verwaltung schuldenbelasteten Codes, dass sie keine Kapazität haben, neue Ansätze oder Technologien zu erkunden.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Technische Schulden erhöhen direkt die Kosten aller Entwicklungsarbeit, da Entwickler Komplexität navigieren, Abkürzungen umgehen und zusätzliche Tests durchführen müssen, um dies zu kompensieren.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Technische Schulden in der Codebasis zwingen Entwickler dazu, übermäßig viel Zeit damit zu verbringen, bestehende Probleme zu umgehen, bevor sie neue Features implementieren.

## Causes ▼

- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die anhaltende Priorisierung sofortiger Lieferung über langfristige Gesundheit durch das Management verhindert, dass dedizierte Zeit oder Budget für die Behebung von Design-Abkürzungen bereitgestellt wird, wodurch sie sich als Schulden anhäufen.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Wenn Code nie verbessert oder umstrukturiert wird, häufen sich Design-Abkürzungen und Komplexität über die Zeit zu erheblichen Schulden an.
- [Zeitdruck](zeitdruck.md)
<br/>  Enge Termine drängen Entwickler dazu, Abkürzungen zu nehmen und Qualitätspraktiken zu überspringen, was direkt technische Schulden schafft.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Peer-Review gelangen schlechte Designentscheidungen und Implementierungsabkürzungen unwidersprochen in die Codebasis.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne Tests ist Refactoring riskant, sodass schuldenbelasteter Code unangetastet bleibt und sich weiter anhäuft.
- [Implementierung beginnt ohne Design](implementierung-beginnt-ohne-design.md)
<br/>  Programmierung ohne vorheriges Design führt zu Ad-hoc-Architektur und Implementierungsabkürzungen, die zu technischen Schulden werden.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Jeder Workaround ist selbst ein Stück technischer Schulden, sodass sich mit der Anhäufung von Workarounds das gesamte Schuldenniveau der Codebasis direkt erhöht.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Aufgeschobene Entscheidungen werden schließlich mit temporären oder Ad-hoc-Lösungen gelöst, die dauerhaft werden und direkt zu den technischen Schulden des Systems beitragen.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Das Erzwingen neuer Anforderungen in eine inkompatible Architektur schafft erhebliche technische Schulden durch kompromittierte Designs.

## Detection Methods ○

- **Codebasis-Metriken:** Beobachtung von Metriken wie zyklomatischer Komplexität, Kopplung und Code-Abdeckung. Hohe Werte deuten oft auf technische Schulden hin.
- **Fehlerverfolgungssysteme:** Analyse der Arten und Häufigkeit von Fehlern, besonders solcher, die bestimmte Module betreffen.
- **Entwicklerbefragungen/-interviews:** Befragung von Entwicklern zu ihren Schmerzpunkten, Bereichen der Codebasis, die sie meiden, und wahrgenommenen technischen Schulden.
- **Code-Audits:** Durchführung regelmäßiger, systematischer Überprüfungen der Codebasis zur Identifikation von Problembereichen.
- **Retrospektiven:** Diskussion wiederkehrender Probleme und Identifikation, ob sie aus technischen Schulden entstehen.
- **Sprint-Planungs-Analyse:** Nachverfolgung, welcher Prozentsatz der Sprint-Kapazität technischen Verbesserungen zugeteilt wird.
- **Geschwindigkeitstrends:** Nachverfolgung, ob die Entwicklungsgeschwindigkeit über die Zeit aufgrund zunehmender technischer Komplexität sinkt.

## Examples
Eine Legacy-E-Commerce-Plattform hat eine stark gekoppelte monolithische Architektur. Das Hinzufügen eines neuen Zahlungs-Gateways erfordert Änderungen über mehrere, scheinbar unzusammenhängende Module hinweg, was zu Wochen der Entwicklung und mehreren neuen Fehlern in Produktion führt. In einem anderen Fall wurde eine Funktion, die ursprünglich für eine einfache Aufgabe entworfen wurde, über die Zeit mit zahlreichen `if-else`-Anweisungen und Sonderfällen modifiziert, was sie Tausende Zeilen lang und unmöglich zu verstehen oder zu testen macht.

Ein Softwareunternehmen hat festgestellt, dass sein Nutzerauthentifizierungssystem auf veralteten Bibliotheken mit bekannten Sicherheitslücken aufgebaut ist. Das Entwicklungsteam schätzt, dass es drei Wochen dauern würde, das Authentifizierungssystem zu modernisieren, was Sicherheit und Wartbarkeit erheblich verbessern würde. Die Produkt-Roadmap ist jedoch für die nächsten sechs Monate mit neuen Features vollgepackt, und das Management weigert sich, Entwicklerzeit für "Infrastrukturarbeit" bereitzustellen, die keinen direkten Kundenwert liefert. Im folgenden Jahr verbringt das Team geschätzt acht Wochen insgesamt damit, Einschränkungen des alten Authentifizierungssystems zu umgehen, sich mit Sicherheitspatches zu befassen und Integrationsprobleme zu beheben, die durch die Modernisierungsarbeit hätten beseitigt werden können.

Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der das Produktkatalog-Modul zu einer 5.000-Zeilen-monolithischen Klasse angewachsen ist, die Stunden braucht, um verstanden und getestet zu werden. Entwickler schätzen häufig zusätzliche Zeit für katalogbezogene Features aufgrund der Komplexität ein, aber Anfragen, das Modul zu refaktorieren, werden immer zugunsten des Hinzufügens neuer Produktfeatures aufgeschoben. Schließlich braucht ein kritischer Fehler im Katalogcode zwei Wochen zur Behebung wegen der Komplexität, was mehr Zeit kostet, als ein ordentliches Refactoring erfordert hätte.
