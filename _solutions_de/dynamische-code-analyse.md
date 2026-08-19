---
title: Dynamische Code-Analyse
description: Prüfung von Sicherheitseigenschaften durch Ausführung und Beobachtung
  des Programmverhaltens.
category:
- Security
- Testing
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- authentication-bypass-vulnerabilities
- memory-leaks
- inadequate-error-handling
- legacy-code-without-tests
- insufficient-testing
layout: solution
lang: de
en_slug: dynamic-code-analysis
related_solutions:
- slug: static-code-analysis
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: penetration-tests
  similarity: 0.8
- slug: fuzz-testing
  similarity: 0.8
- slug: security-tests
  similarity: 0.75
- slug: regression-tests
  similarity: 0.75
---

## Description

Dynamische Code-Analyse testet die Sicherheitseigenschaften eines Systems, indem es tatsächlich ausgeführt und sein Laufzeitverhalten beobachtet wird — präparierte Eingaben senden, Speicherverhalten überwachen oder internen Datenfluss instrumentieren —, statt über Schwachstellen allein aus dem Quellcode heraus zu argumentieren. Diese Unterscheidung ist für Legacy-Systeme besonders wichtig, da statische Überprüfung alten Codes häufig unvollständig oder unmöglich ist: Quellcode könnte teilweise verloren sein, in speicherunsicheren Sprachen geschrieben, deren Fehlermodi sich nur unter spezifischen Laufzeitbedingungen manifestieren, oder zu sehr mit Laufzeitkonfiguration und Drittanbieter-Abhängigkeiten verwoben, als dass ein statischer Prüfer sicher darüber argumentieren könnte. Werkzeuge in dieser Kategorie reichen von Dynamic Application Security Testing, das eine laufende Anwendung so untersucht, wie es ein externer Angreifer täte, über Interactive Application Security Testing, das internen Datenfluss während der Ausführung beobachtet, bis zu Speicheranalysatoren wie Valgrind oder AddressSanitizer, die Pufferüberläufe und Use-after-Free-Fehler erkennen, die für jede Quellcode-Überprüfung unsichtbar sind. Weil dynamische Analyse nur die Codepfade ausübt, die sie tatsächlich auslöst, ergänzt sie statische Analyse und manuelle Überprüfung, statt sie zu ersetzen — ein statischer Scan könnte verdächtige String-Verkettung als Codequalitätsbedenken markieren, aber nur ein dynamischer Test kann bestätigen, ob dieses Muster in der Praxis tatsächlich ausnutzbar ist. Dies ist speziell für die Modernisierung von Legacy-Systemen wichtig, weil es einem Team erlaubt festzustellen, welche von vielen langjährigen, zuvor unverifizierten Risikosignalen in altem Code echte, derzeit ausnutzbare Schwachstellen darstellen, die eine Priorisierung wert sind, gegenüber theoretischen Schwächen, die in der Praxis nie erreichbar waren.

## How to Apply ◆

> Legacy-Systeme können oft nicht allein durch statische Code-Überprüfung vollständig analysiert werden, wegen komplexen Laufzeitverhaltens, dynamischer Konfiguration und Drittanbieter-Abhängigkeiten. Dynamische Analyse testet Sicherheitseigenschaften, indem die Anwendung ausgeführt und ihr Verhalten unter verschiedenen Bedingungen beobachtet wird.

- Setzen Sie Dynamic-Application-Security-Testing(DAST)-Werkzeuge ein, die mit der laufenden Legacy-Anwendung so interagieren, wie es ein externer Angreifer täte — bösartige Payloads senden, Authentifizierungsumgehungen testen und über die Schnittstellen der Anwendung nach Injection-Schwachstellen suchen.
- Implementieren Sie Interactive Application Security Testing (IAST), indem die Laufzeitumgebung der Legacy-Anwendung instrumentiert wird, um zu beobachten, wie sie Eingaben intern behandelt. IAST erkennt Schwachstellen, die DAST möglicherweise verpasst, indem Datenfluss durch den Anwendungscode verfolgt wird.
- Führen Sie Speicheranalyse-Werkzeuge (Valgrind, AddressSanitizer oder plattformspezifische Äquivalente) gegen Legacy-Anwendungen aus, die in speicherunsicheren Sprachen (C, C++) geschrieben sind, um Pufferüberläufe, Use-after-Free-Fehler und Speicherlecks zu erkennen, die Sicherheitslücken erzeugen.
- Konfigurieren Sie dynamische Analyse so, dass sie gegen eine Staging-Umgebung läuft, die Produktion widerspiegelt, mittels realistischer Testdaten und Konfiguration. Testing gegen eine minimale Entwicklungsumgebung könnte Schwachstellen verpassen, die sich nur unter produktionsähnlichen Bedingungen manifestieren.
- Integrieren Sie dynamisches Sicherheitstesting in die CI/CD-Pipeline, sodass neue Deployments automatisch gescannt werden, bevor sie Produktion erreichen. Beginnen Sie mit einer fokussierten Menge hochpriorisierter Tests, um die Pipeline-Ausführungszeit handhabbar zu halten.
- Ergänzen Sie automatisierte dynamische Analyse durch manuelles exploratives Sicherheitstesting für komplexe Geschäftslogikschwachstellen, die automatisierte Werkzeuge nicht erkennen können, wie Autorisierungsumgehungen in mehrstufigen Workflows.

## Tradeoffs ⇄

> Dynamische Analyse erkennt Laufzeit-Sicherheitsschwachstellen, die statische Analyse nicht finden kann, erfordert aber eine laufende Anwendungsumgebung und erreicht möglicherweise keine vollständige Codeabdeckung.

**Vorteile:**

- Entdeckt Schwachstellen im tatsächlichen Verhalten der laufenden Anwendung, einschließlich Problemen, die durch Laufzeitkonfiguration, Drittanbieter-Bibliotheken und Umgebungsfaktoren verursacht werden.
- Testet die Anwendung so, wie ein Angreifer mit ihr interagieren würde, und findet Schwachstellen, die tatsächlich ausnutzbar sind, statt theoretisch.
- Erkennt Speichersicherheitsprobleme in nativem Code, die für Quellcode-Überprüfung unsichtbar sind.
- Erfordert keinen Zugriff auf Quellcode, was es auf Legacy-Systeme mit verlorenem oder nicht verfügbarem Quellcode anwendbar macht.

**Kosten und Risiken:**

- Dynamische Analyse kann nur Codepfade testen, die während des Tests tatsächlich ausgeführt werden, was potenziell Schwachstellen in ungetesteten Pfaden verpasst.
- Das Ausführen von Sicherheitstests gegen produktionsähnliche Umgebungen erfordert sorgfältige Isolation, um zu verhindern, dass Testaktivitäten echte Daten oder Services beeinflussen.
- DAST-Werkzeuge können erhebliche Last erzeugen und Sicherheitskontrollen (WAF, IDS) auslösen, die das Testing stören.
- Legacy-Systeme könnten fragil sein und unter den ungewöhnlichen Eingaben, die dynamische Analyse erzeugt, abstürzen oder unvorhersehbar reagieren.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie dynamische Code-Analyse Sicherheitsschwachstellen in Legacy-Systemen aufdeckt.

Eine Legacy-Java-Webanwendung ist seit 12 Jahren mit minimalem Sicherheitstesting in Produktion. Das Team setzt einen DAST-Scanner gegen eine Staging-Instanz ein und entdeckt 14 reflektierte XSS-Schwachstellen, 3 SQL-Injection-Punkte und eine Authentifizierungsumgehung im Passwort-Zurücksetzen-Ablauf. Die SQL-Injection-Schwachstellen liegen in Legacy-JSP-Seiten, die Abfragen mittels String-Verkettung konstruieren. Die Authentifizierungsumgehung tritt auf, weil das Passwort-Zurücksetzen-Token eine vorhersehbare sequenzielle Zahl statt eines kryptografischen Zufallswerts ist. Statische Analyse hatte die String-Verkettung zuvor als Codequalitätsproblem markiert, konnte aber die Ausnutzbarkeit nicht bestätigen — der dynamische Test beweist, dass dies aktiv ausnutzbare Schwachstellen sind, und priorisiert die Behebung nach Schweregrad.

Ein Legacy-C++-Handelssystem verarbeitet Marktdaten durch eine hochperformante Parsing-Pipeline. Das Team führt die Anwendung während ihrer Integrationstest-Suite unter AddressSanitizer aus und entdeckt einen Heap-Pufferüberlauf im Marktdaten-Parser, der auftritt, wenn ein bestimmtes fehlerhaftes Nachrichtenformat verarbeitet wird. Der Überlauf erlaubt einem Angreifer, der präparierte Marktdatennachrichten injizieren kann, beliebigen Code auf dem Handelsserver auszuführen. Diese Schwachstelle existierte 7 Jahre, wurde aber nie durch normale Marktdaten ausgelöst, was sie für funktionales Testing und Code-Review unsichtbar machte. Der dynamische Analysefund führt zu einer gezielten Korrektur der Grenzprüfung im Parser, und das Team fügt die fehlerhafte Nachricht als dauerhaften Regressionstestfall hinzu.
