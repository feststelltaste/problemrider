---
title: Notfallübungen
description: Training von Verhalten bei Sicherheitsvorfällen und Testen von Notfallprozessen.
category:
- Security
- Operations
problems:
- slow-incident-resolution
- constant-firefighting
- system-outages
- monitoring-gaps
- poor-operational-concept
- knowledge-gaps
- poorly-defined-responsibilities
- missing-rollback-strategy
layout: solution
lang: de
en_slug: emergency-drills
related_solutions:
- slug: incident-response-measures
  similarity: 0.8
- slug: backup-and-recovery
  similarity: 0.75
- slug: security-incident-handling
  similarity: 0.75
- slug: runbooks
  similarity: 0.7
- slug: incident-management
  similarity: 0.7
- slug: security-training
  similarity: 0.7
---

## Description

Notfallübungen sind geübte Simulationen von Sicherheitsvorfällen und operativen Notfällen — von verbalen Tabletop-Übungen bis zu vollständig simulierten Vorfällen in Nicht-Produktionsumgebungen — durchgeführt speziell um zu testen, ob die Vorfallreaktionsprozeduren, das Tooling und das Personal einer Organisation tatsächlich funktionieren, bevor eine echte Krise die Frage erzwingt. Legacy-Systeme sind hier unverhältnismäßig gefährdet, weil ihre Reaktionsprozeduren, falls sie überhaupt existieren, häufig undokumentiert sind, einmal geschrieben und nie überarbeitet, oder vom stillschweigenden Wissen bestimmter Personen abhängen, die möglicherweise nicht mehr erreichbar oder sogar noch beschäftigt sind, wenn ein Vorfall eintritt. Eine Übung deckt genau diese Art von Verfall unter risikoarmen, kontrollierten Bedingungen auf: veraltete Eskalationskontaktlisten, Runbooks, die auf nicht mehr existierende Infrastruktur verweisen, oder Backup-Wiederherstellungsprozeduren, von denen angenommen wurde, dass sie funktionieren, die aber nie tatsächlich Ende-zu-Ende geübt wurden. Weil dieser Verfall still ist und sich über die Zeit summiert — Infrastruktur ändert sich weiter, während ein ungeprobtes Runbook eingefroren bleibt bei dem, was es beschrieb, als es geschrieben wurde —, müssen Übungen in regelmäßigem Takt wiederholt und über Szenarien und Teilnehmer rotiert werden, statt einmal ausgeführt und als erledigt betrachtet zu werden. Ihr Wert wird speziell realisiert, wenn Befunde bis zur Lösung verfolgt und in nachfolgenden Übungen erneut getestet werden, da eine einzelne Übung, die eine Lücke offenbart, ohne einen systematischen Nachverfolgungsmechanismus nur das Problem dokumentiert, statt es zu beheben.

## How to Apply ◆

> Legacy-Systeme sind während Sicherheitsvorfällen besonders verwundbar, weil Reaktionsprozeduren oft undokumentiert, ungetestet und von Personen abhängig sind, die möglicherweise nicht verfügbar sind. Notfallübungen bauen organisatorisches Muskelgedächtnis für Vorfallreaktion auf, bevor eine echte Krise eintritt.

- Definieren Sie Vorfallreaktionsszenarien basierend auf dem tatsächlichen Risikoprofil des Legacy-Systems: Datenschutzverletzung, Ransomware-Infektion, Denial of Service, kompromittierte Anmeldedaten, unautorisierter Datenzugriff und kritische Schwachstellenveröffentlichung. Nutzen Sie vergangene Vorfälle und Beinaheunfälle als Grundlage für Szenarien.
- Führen Sie Tabletop-Übungen durch, bei denen das Vorfallreaktionsteam ein Szenario verbal durchgeht und bespricht, wer was tut, welche Werkzeuge genutzt werden und welche Informationen benötigt werden. Dieses günstige Format offenbart Kommunikationslücken und unklare Verantwortlichkeiten, ohne Produktionssysteme zu beeinflussen.
- Führen Sie simulierte Vorfälle in Nicht-Produktionsumgebungen durch, in denen das Team tatsächlich Reaktionsprozeduren ausführen muss: betroffene Systeme isolieren, forensische Beweise sammeln, mit Stakeholdern kommunizieren und aus Backups wiederherstellen. Zeitmessen Sie die Übungen, um Baseline-Reaktionsfähigkeiten zu etablieren.
- Testen Sie Backup-Wiederherstellung als Teil jeder Übung. Zu verifizieren, dass Backups existieren, reicht nicht — das Team muss demonstrieren, dass es das Legacy-System innerhalb des definierten Recovery Time Objective in einen funktionsfähigen Zustand wiederherstellen kann.
- Rotieren Sie Übungsteilnehmer, sodass Vorfallreaktionsfähigkeit nicht auf wenige Personen konzentriert ist. Stellen Sie sicher, dass Bereitschaftsingenieure, Manager, Kommunikationspersonal und Rechtskontakte alle an für ihre Rollen relevanten Übungen teilnehmen.
- Dokumentieren Sie gelernte Lektionen aus jeder Übung und verfolgen Sie die Lösung identifizierter Lücken. Pflegen Sie eine laufende Liste von Verbesserungspunkten und verifizieren Sie deren Umsetzung in nachfolgenden Übungen.
- Planen Sie Übungen in regelmäßigen Abständen (vierteljährlich wird empfohlen) und variieren Sie die Szenarien, um unterschiedliche Vorfalltypen abzudecken und sicherzustellen, dass Reaktionsfähigkeiten nicht verkümmern.

## Tradeoffs ⇄

> Notfallübungen bauen zuverlässige Vorfallreaktionsfähigkeit auf und identifizieren Lücken, bevor echte Vorfälle sie ausnutzen, erfordern aber Zeitinvestition von mehreren Teams und können störend sein.

**Vorteile:**

- Deckt Lücken in Vorfallreaktionsprozeduren, Tooling und Personal auf, bevor ein echter Vorfall sie unter Druck offenlegt.
- Baut Teamvertrauen auf und reduziert Panik während echter Vorfälle, indem geübte, vertraute Reaktionsmuster bereitgestellt werden.
- Testet Backup- und Wiederherstellungsprozeduren unter realistischen Bedingungen und stellt sicher, dass sie bei Bedarf tatsächlich funktionieren.
- Identifiziert unklare Verantwortlichkeiten und Kommunikationswege, die während echter Vorfälle Verzögerungen verursachen.

**Kosten und Risiken:**

- Übungen verbrauchen Engineering-Zeit, die für Entwicklung oder Betrieb aufgewendet werden könnte, was Managementunterstützung zur Priorisierung erfordert.
- Schlecht gestaltete Übungen, die unrealistisch oder zu einfach sind, bieten falsches Vertrauen, ohne echte Fähigkeit aufzubauen.
- Übungen, die mit produktionsnahen Systemen interagieren, tragen ein kleines Risiko, unbeabsichtigte Auswirkung zu verursachen, wenn die Isolation unvollständig ist.
- Übungsmüdigkeit kann entstehen, wenn Übungen zu häufig oder repetitiv sind, was Engagement und Lernen reduziert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Notfallübungen die Vorfallreaktion für Legacy-Systeme verbessern.

Ein Legacy-Zahlungsverarbeitungssystem erlebt eine vermutete Datenschutzverletzung. Das Vorfallreaktionsteam verbringt 4 Stunden damit zu bestimmen, wer die Autorität hat, das System offline zu nehmen, weitere 2 Stunden damit, die Datenbank-Backup-Anmeldedaten zu finden (die in einer Tabellenkalkulation auf dem archivierten Laufwerk eines ehemaligen Mitarbeiters gespeichert sind), und entdeckt, dass das jüngste wiederherstellbare Backup 72 Stunden alt ist — was das im Business-Continuity-Plan definierte 24-Stunden-RPO überschreitet. Eine Nachvorfallüberprüfung führt zu vierteljährlichen Notfallübungen. Die erste Übung offenbart, dass die Eskalationskontaktliste 18 Monate veraltet ist und dass drei der sieben Vorfallreaktionsteammitglieder nie eine Produktionsdatenbankwiederherstellung durchgeführt haben. Über vier vierteljährliche Übungen reduziert das Team seine simulierte Reaktionszeit von 6 Stunden auf 90 Minuten, etabliert eine aktuelle Eskalationsmatrix mit automatisierter Alarmierung und verifiziert, dass Backups innerhalb des 4-Stunden-RTO wiederherstellbar sind.

Das Betriebsteam eines Legacy-Gesundheitssystems hat ein Runbook für Sicherheitsvorfälle, aber es wurde vor fünf Jahren geschrieben und nie getestet. Während einer Übung, die eine Ransomware-Infektion simuliert, entdeckt das Team, dass das Runbook auf nicht mehr existierende Netzwerksegmente verweist, Isolationsprozeduren für eine ersetzte Firewall spezifiziert und die drei neuen Microservices auslässt, die zur Architektur des Legacy-Systems hinzugefügt wurden. Die Übung dauert 5 statt der erwarteten 2 Stunden, weil Teammitglieder um die veralteten Prozeduren herum improvisieren müssen. Die Übung führt zu einer vollständigen Runbook-Neuschreibung, der Erstellung automatisierter Isolationsskripte, die mit der aktuellen Infrastruktur funktionieren, und der Hinzufügung eines Runbook-Überprüfungsschritts zu jedem Infrastrukturänderungsprozess. Die nachfolgende Übung wird in 2,5 Stunden ohne erforderliche Improvisation abgeschlossen.
