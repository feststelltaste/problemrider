---
title: Web Application Firewall
description: Filterung von HTTP-Traffic auf Anwendungsebene gegen
  Web-Angriffe.
category:
- Security
- Operations
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- rate-limiting-issues
- system-outages
- legacy-code-without-tests
layout: solution
lang: de
en_slug: web-application-firewall
related_solutions:
- slug: security-monitoring
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: honeypots
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
- slug: secure-programming-interfaces
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.7
---

## Description

Eine Web Application Firewall ist eine Filterschicht, die vor einer Webanwendung platziert wird und eingehenden HTTP-Traffic inspiziert und Anfragen blockiert, die bekannten Angriffsmustern entsprechen, wie SQL-Injection-Payloads, Cross-Site-Scripting-Versuchen oder fehlgeformtem Authentifizierungs-Traffic, bevor diese Anfragen überhaupt den Anwendungscode erreichen. Sie operiert als Reverse-Proxy oder Inline-Appliance, bewertet jede Anfrage gegen einen Regelsatz — typischerweise abgeleitet aus den OWASP Top 10 und verfeinert mit anwendungsspezifischen Mustern — und lässt sie entweder durch, blockiert sie oder markiert sie zur Überprüfung. Für Legacy-Systeme zählt dieser Mechanismus, weil der zugrunde liegende Anwendungscode oft mit Schwachstellen durchsetzt ist, die teuer und riskant direkt zu beheben sind: rohe SQL-Verkettung, verstreut über Tausende von Aufrufstellen, unescapte Ausgabe in Templates, die geschrieben wurden, bevor sichere Codierungspraktiken Standard waren, oder Authentifizierungslogik, die zu brüchig ist, um sie ohne umfangreiches Regressionstesting anzufassen. Eine WAF bietet eine kompensierende Kontrolle, die die ausnutzbare Angriffsfläche sofort verkleinert, ohne irgendeine Änderung an der Legacy-Codebasis selbst zu erfordern, und kauft die Zeit, die benötigt wird, um die tatsächlichen Schwachstellen sicher zu sanieren. Sie wird am Netzwerkrand statt innerhalb der Anwendung eingesetzt, was sie zu einer der wenigen Sicherheitsmaßnahmen macht, die auf Legacy-Systeme angewendet werden können, ohne Zugang zum Quellcode oder Appetit auf Neubereitstellungsrisiko. Da sie auf Musterabgleich statt auf der Behebung des zugrunde liegenden Fehlers beruht, wird sie am besten als mildernder Schild statt als Heilmittel verstanden, und ihre Wirksamkeit muss kontinuierlich sowohl gegen falsch positive Ergebnisse als auch sich entwickelnde Angreifertechniken abgestimmt werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Setzen Sie eine WAF vor Legacy-Webanwendungen als Schutzschicht ein, die keine Anwendungscodeänderungen erfordert
- Beginnen Sie im Monitoring-Modus, um Traffic-Muster zu verstehen und legitime Anfragen als Baseline zu erfassen, bevor Sie Blockierung aktivieren
- Konfigurieren Sie Regeln, die auf die für die Legacy-Anwendung relevantesten OWASP-Top-10-Schwachstellenkategorien abzielen
- Erstellen Sie maßgeschneiderte Regeln für anwendungsspezifische Angriffsmuster, die durch Penetrationstests oder Vorfallanalyse entdeckt wurden
- Implementieren Sie Rate Limiting und Bot-Erkennung, um Legacy-Anwendungen vor Missbrauch und Denial-of-Service-Angriffen zu schützen
- Integrieren Sie WAF-Logs mit dem Sicherheitsmonitoring-System zur Korrelation mit anderen Sicherheitsereignissen
- Überprüfen und stimmen Sie WAF-Regeln regelmäßig ab, um Schutz gegen Falsch-positiv-Raten abzuwägen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet sofortigen Schutz für Legacy-Anwendungen, ohne Codeänderungen zu erfordern
- Fungiert als kompensierende Kontrolle für Schwachstellen, die nicht schnell im Legacy-Code behoben werden können
- Bietet Sichtbarkeit in Angriffsversuche und -muster, die auf die Anwendung abzielen
- Kann relativ schnell im Vergleich zur benötigten Zeit zur Behebung zugrunde liegender Anwendungsschwachstellen eingesetzt werden

**Kosten und Risiken:**
- WAFs können von raffinierten Angreifern umgangen werden, die Payloads gestalten, um Erkennungsregeln zu umgehen
- Falsch positive Ergebnisse können legitimen Traffic blockieren und nutzerseitige Probleme schaffen
- WAFs fügen jeder Anfrage Latenz hinzu, was performance-sensible Legacy-Anwendungen beeinträchtigen kann
- Übermäßiges Vertrauen auf WAFs als Ersatz für die Behebung zugrunde liegender Schwachstellen schafft ein falsches Sicherheitsgefühl
- WAF-Regeln erfordern laufende Abstimmung und Pflege, während sich Angriffstechniken weiterentwickeln

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Reisebuchungsunternehmen entdeckte mehrere SQL-Injection-Schwachstellen in seiner Legacy-Buchungs-Engine während eines Penetrationstests. Die Behebung der Schwachstellen in der 12 Jahre alten Codebasis wurde auf drei Monate Arbeit geschätzt, aufgrund der tief eingebetteten rohen SQL-Muster. Das Team setzte innerhalb einer Woche eine cloudbasierte WAF ein, konfiguriert mit SQL-Injection-Erkennungsregeln, und begann sofort, Ausnutzungsversuche zu blockieren. WAF-Logs zeigten allein im ersten Monat über 500 blockierte SQL-Injection-Versuche. Die WAF diente als Schutzschicht, während das Entwicklungsteam methodisch rohe SQL-Abfragen im folgenden Quartal durch parametrisierte Statements ersetzte.
