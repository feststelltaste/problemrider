---
title: Erhöhter manueller Testaufwand
description: Eine unverhältnismäßige Menge an Zeit wird für manuelles Testen aufgewendet,
  aufgrund fehlender Automatisierung.
category:
- Process
- Testing
related_problems:
- slug: increased-manual-work
  similarity: 0.75
- slug: long-release-cycles
  similarity: 0.7
- slug: manual-deployment-processes
  similarity: 0.65
- slug: inadequate-test-infrastructure
  similarity: 0.65
- slug: insufficient-testing
  similarity: 0.65
- slug: testing-complexity
  similarity: 0.6
solutions:
- test-coverage-strategy
- automated-tests
- characterization-tests
- regression-testing
- acceptance-tests
- ci-cd-pipeline
- smoke-testing
- production-like-test-data
- exploratory-testing
layout: problem
lang: de
en_slug: increased-manual-testing-effort
---

## Description

Erhöhter manueller Testaufwand tritt auf, wenn Teams übermäßig viel Zeit mit manuellen Verifikationsaktivitäten verbringen, weil automatisiertes Testen unzureichend ist oder fehlt. Während manuelles Testen wertvoll sein kann, besonders für Nutzererlebnis und exploratives Testen, schafft übermäßige Abhängigkeit von manuellen Prozessen Engpässe, Inkonsistenz und Skalierbarkeitsprobleme. Manuelles Testen wird zu einem limitierenden Faktor für Release-Häufigkeit und Teamproduktivität, wenn es genutzt wird, um unzureichende Automatisierung zu kompensieren.

## Indicators ⟡
- Erhebliche Teile jedes Release-Zyklus sind manuellen Testaktivitäten gewidmet
- Das Test-Team oder Entwickler verbringen die meiste Zeit mit der Ausführung repetitiver manueller Testfälle
- Release-Zeitpläne werden durch manuelle Testkapazität statt durch Entwicklungsabschluss begrenzt
- Dieselben manuellen Tests werden für jedes Release oder jede Änderung wiederholt ausgeführt
- Manuelles Testen entdeckt Fehler, die von automatisierten Tests hätten erfasst werden sollen

## Symptoms ▲

- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Umfangreiches manuelles Testen schafft Engpässe, die Releases verzögern und häufige Deployments verhindern.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Teammitglieder, die Zeit mit manuellem Testen verbringen, haben weniger Kapazität für Entwicklungsarbeit, was die Gesamtgeschwindigkeit verlangsamt.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Manuelles Testen erfordert erhebliche menschliche Ressourcen, die besser für Entwicklung genutzt werden könnten, was die Gesamtkosten erhöht.
- [Inkonsistente Ausführung](inkonsistente-ausfuehrung.md)
<br/>  Menschliche Tester führen Tests unweigerlich unterschiedlich aus, was zu inkonsistenter Abdeckung und übersehenen Defekten führt.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Repetitive manuelle Testaufgaben sind demotivierend und laugen Entwicklerenergie und -begeisterung aus.

## Causes ▼

- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende Werkzeuge, Umgebungen oder Automatisierungs-Frameworks zwingen Teams, sich auf manuelles Testen zu verlassen.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Legacy-Systeme ohne automatisierte Tests erfordern manuelle Verifikation für jede Änderung.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Eng gekoppelter oder schlecht strukturierter Code macht Automatisierung schwierig, was Teams zwingt, manuell zu testen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden erschweren Investitionen in Testautomatisierung, was manuelles Testen fortbestehen lässt.

## Detection Methods ○
- **Testzeitanalyse:** Nachverfolgung, welcher Prozentsatz der Release-Zyklus-Zeit für manuelles vs. automatisiertes Testen aufgewendet wird
- **Testausführungs-Tracking:** Beobachtung, wie viele Testfälle manuell vs. automatisiert ausgeführt werden
- **Ressourcenzuteilung:** Messung der menschlichen Ressourcen, die manuellen Testaktivitäten gewidmet sind
- **Release-Engpassanalyse:** Identifikation, ob manuelles Testen Releases mehr verzögert als Entwicklungsarbeit
- **Bewertung der Testabdeckung:** Vergleich der manuellen Testabdeckung mit der automatisierten Testabdeckung

## Examples

Ein Webanwendungs-Team hat eine umfassende Suite manueller Testfälle, die Nutzerregistrierung, Login, Profilverwaltung, Content-Erstellung und administrative Funktionen abdeckt. Vor jedem zweiwöchentlichen Release verbringen zwei Teammitglieder drei volle Tage damit, 200+ manuelle Testfälle auszuführen und durch die Anwendung zu klicken, um zu verifizieren, dass bestehende Funktionalität noch funktioniert. Diese manuellen Regressionstests verbrauchen 30 % der Teamkapazität und verhindern häufigere Releases. Als automatisiertes Testen schließlich für die Kern-Nutzer-Flows implementiert wird, wird die manuelle Testzeit auf einen halben Tag reduziert, der sich auf exploratives Testen und neue Features konzentriert, was dem Team erlaubt, wöchentlich statt zweiwöchentlich zu releasen. Ein weiteres Beispiel betrifft eine mobile Banking-Anwendung, bei der regulatorische Compliance umfangreiches Testen von Finanztransaktionen, Sicherheitsfeatures und Datenhandhabung erfordert. Das Team verbringt zwei Wochen manuellen Testens für jedes Release, wobei Tester manuell Konten erstellen, Transaktionen durchführen, Berichte erzeugen und Berechnungen verifizieren. Das manuelle Testen ist nicht nur zeitaufwendig, sondern auch fehleranfällig, da menschliche Tester gelegentlich Randfälle übersehen oder Fehler bei der Verifikation machen. Die Implementierung automatisierten Testens für die Kern-Finanzberechnungen und Transaktionsflüsse reduziert die manuelle Testlast um 70 %, während sie Testabdeckung und Zuverlässigkeit verbessert.
