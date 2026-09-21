---
title: Zunehmende Brüchigkeit
description: Softwaresysteme werden über die Zeit brüchiger und anfälliger für Ausfälle,
  wobei kleine Änderungen unvorhersehbare und weitreichende Auswirkungen haben.
category:
- Architecture
- Code
- Management
related_problems:
- slug: brittle-codebase
  similarity: 0.75
- slug: quality-degradation
  similarity: 0.7
- slug: maintenance-cost-increase
  similarity: 0.65
- slug: gradual-performance-degradation
  similarity: 0.65
- slug: testing-environment-fragility
  similarity: 0.65
- slug: increased-bug-count
  similarity: 0.65
solutions:
- incremental-refactoring
- technical-debt-backlog
- dependency-pinning
- code-hotspot-analysis
- improvement-budget
- mikado-method
- preparatory-refactoring
- characterization-tests
- change-impact-analysis
- defect-triage-process
- baseline-measurement
- cost-of-delay
- no-regret-moves
- risk-quantification
- technical-debt-assessment
- debt-classification
- quality-ratchet
- debt-accrual-analysis
- continuous-dependency-updates
- automated-code-migration
layout: problem
lang: de
en_slug: increasing-brittleness
---

## Description

Zunehmende Brüchigkeit tritt auf, wenn Softwaresysteme über die Zeit fortschreitend brüchiger und instabiler werden, wobei scheinbar geringfügige Änderungen unerwartete Ausfälle verursachen oder unzusammenhängende Funktionalität brechen können. Diese Brüchigkeit entwickelt sich, während sich technische Schulden anhäufen, Abhängigkeiten komplexer werden und sich die Systemarchitektur ohne ordentliche Wartung verschlechtert. Brüchige Systeme sind schwer sicher zu modifizieren und zeigen oft unvorhersehbares Verhalten.

## Indicators ⟡

- Kleine Änderungen verursachen häufig unerwartete Ausfälle in unzusammenhängenden Systembereichen
- Die Anzahl der Fehler steigt, selbst wenn keine neuen Features hinzugefügt werden
- Das Systemverhalten wird zunehmend unvorhersehbar
- Mehr Zeit wird mit Debugging verbracht als mit der Entwicklung neuer Funktionalität
- Änderungen, die in der Entwicklung funktionierten, schlagen in Produktion aus unklaren Gründen fehl

## Symptoms ▲

- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wenn kleine Änderungen unvorhersehbare Ausfälle verursachen, bekommen Entwickler Angst, das System zu ändern.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Ein brüchiges System zeigt unerwartetes Verhalten, wenn Änderungen vorgenommen werden, was Ergebnisse schwer vorhersehbar macht.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Brüchige Systeme erzeugen mehr Fehler, während Änderungen durch eng gekoppelte Komponenten kaskadieren.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Entwickler müssen in brüchigen Systemen vorsichtig vorgehen und jede Änderung umfangreich testen, was die Entwicklung verlangsamt.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Häufige unerwartete Ausfälle durch Brüchigkeit halten Teams im reaktiven Modus, während sie auf kaskadierende Probleme reagieren.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Eine brüchige Codebasis, in der kleine Änderungen unvorhersehbare Auswirkungen haben, führt direkt dazu, dass Entwickler große Schätzungen abgeben.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten bedeuten, dass sich Änderungen unvorhersehbar durch das System fortpflanzen, was es brüchig macht.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden verschlechtern die Systemarchitektur über die Zeit, was das System fortschreitend brüchiger macht.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Workarounds schaffen versteckte Abhängigkeiten und umgehen entworfene Schnittstellen, was das System anfällig für Ausfälle macht.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne umfassende Tests bleiben Regressionen unentdeckt, während Änderungen vorgenommen werden, was Brüchigkeit wachsen lässt.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Minderwertiger Code mit unklaren Verträgen und schlechter Struktur wird zunehmend brüchig, während er sich weiterentwickelt.

## Detection Methods ○

- **Ausfallraten-Tracking:** Beobachtung der Häufigkeit von Systemausfällen und ihrer Beziehung zu kürzlichen Änderungen
- **Änderungsauswirkungsanalyse:** Bewertung, wie oft Änderungen unzusammenhängende Systembereiche betreffen
- **Fehlertrend-Analyse:** Nachverfolgung von Fehlerberichten über die Zeit, besonders Regressionsfehlern
- **Systemstabilitätsmetriken:** Messung von Systemverfügbarkeit, Fehlerraten und Performance-Konsistenz
- **Änderungsrisikobewertung:** Bewertung des wahrgenommenen Risikos, das mit Systemmodifikationen verbunden ist

## Examples

Eine E-Commerce-Plattform erlebt einen kritischen Ausfall in ihrer Produktempfehlungs-Engine nach einer scheinbar unzusammenhängenden Änderung am Nutzerauthentifizierungssystem. Die Untersuchung zeigt, dass die Authentifizierungsänderung eine gemeinsam genutzte Caching-Schicht modifizierte, auf die sich die Empfehlungs-Engine verließ, obwohl diese Abhängigkeit nirgendwo dokumentiert war. Diese Art unerwarteten Ausfalls passiert zunehmend häufiger – eine Datenbankschema-Änderung bricht das Reporting-System, ein UI-Update verursacht Checkout-Ausfälle, und eine Performance-Optimierung löst Fehler in der Bestandsverfolgung aus. Das Entwicklungsteam verbringt mehr Zeit mit der Untersuchung und Behebung dieser Kaskadenausfälle als mit der Implementierung neuer Features. Ein weiteres Beispiel betrifft ein Finanzhandelssystem, bei dem das Hinzufügen einer neuen Datenvalidierungsregel dazu führt, dass bestehende Trades aufgrund subtiler Änderungen im Timing des Datenflusses nicht mehr verarbeitet werden. Das System ist so miteinander verbunden und brüchig geworden, dass jede Änderung riskiert, Ausfälle in entfernten Teilen des Systems auszulösen, was die Entwicklung extrem langsam und riskant macht.
