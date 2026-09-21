---
title: Digitale Forensik
description: Etablierung von Methoden zur Untersuchung von Sicherheitsvorfällen und
  Straftaten.
category:
- Security
problems:
- insufficient-audit-logging
- debugging-difficulties
- slow-incident-resolution
- monitoring-gaps
- data-protection-risk
- silent-data-corruption
layout: solution
lang: de
en_slug: digital-forensics
related_solutions:
- slug: incident-response-measures
  similarity: 0.75
- slug: logging-and-monitoring
  similarity: 0.75
- slug: audit-trail-management
  similarity: 0.75
- slug: security-monitoring
  similarity: 0.7
- slug: endpoint-detection-and-response
  similarity: 0.7
- slug: honeypots
  similarity: 0.7
---

## Description

Digitale Forensik ist im Kontext des Legacy-Systembetriebs die Praxis, proaktiv das Logging, die Beweiserhaltung und die Untersuchungsprozeduren zu etablieren, die nötig sind, um exakt zu rekonstruieren, was während eines Sicherheitsvorfalls geschah — wer worauf zugriff, wann, wie und mit welcher Wirkung. Es ist eine Bereitschaftsdisziplin statt einer reaktiven: Umfassendes Logging, manipulationssichere Log-Aggregation, synchronisierte Zeitstempel und definierte Chain-of-Custody-Prozeduren müssen alle existieren, bevor ein Vorfall eintritt, weil sie nicht nachträglich in einen bereits stattgefundenen Angriff eingebaut werden können. Legacy-Systeme sind hier besonders exponiert, da sie häufig in Ären mit schwächeren Logging-Konventionen gebaut wurden, und operative Gewohnheiten wie aggressive Log-Rotation zur Speicherplatzeinsparung zerstören routinemäßig genau die Beweise, die eine Untersuchung bräuchte. Ohne diese Grundlage kann ein vermuteter Einbruch in ein Legacy-System oft überhaupt nicht aufgeklärt werden — die Zeitlinie kann nicht rekonstruiert werden, der Umfang kann nicht eingegrenzt werden, und weder rechtliche Schritte noch regulatorische Meldung können auf soliden Beweisen fortschreiten. Die Etablierung digitaler Forensik-Fähigkeit geht daher weniger darum, auf einen spezifischen Angriff zu reagieren, und mehr darum sicherzustellen, dass welcher Vorfall auch immer schließlich in einem alternden, unzureichend instrumentierten System eintritt, eine Spur hinterlässt, der Ermittler tatsächlich folgen können.

## How to Apply ◆

> Legacy-Systeme zerstören oft die Beweise, die zur Untersuchung von Sicherheitsvorfällen nötig sind, oder sammeln sie gar nicht erst. Digitale-Forensik-Bereitschaft stellt sicher, dass bei Vorfällen ausreichend Beweise existieren und zuverlässig gesammelt, erhalten und analysiert werden können.

- Aktivieren Sie umfassendes Logging für alle sicherheitsrelevanten Ereignisse: Authentifizierung, Autorisierungsentscheidungen, Datenzugriff, administrative Aktionen, Konfigurationsänderungen und Netzwerkverbindungen. Stellen Sie sicher, dass Logs genug Detail für die Rekonstruktion von Vorfallzeitlinien enthalten.
- Implementieren Sie zentralisierte, manipulationssichere Log-Aggregation, die Logs von allen Legacy-System-Komponenten an einen gesicherten, Append-Only-Log-Speicher nahezu in Echtzeit weiterleitet. Dies bewahrt Beweise, selbst wenn ein Angreifer Logs auf einzelnen Systemen kompromittiert und löscht.
- Definieren Sie Beweiserhaltungsprozeduren: wie Festplatten-Images, Speicherabbilder, Netzwerk-Paketaufzeichnungen und Log-Snapshots erfasst werden, ohne die ursprünglichen Beweise zu verändern. Dokumentieren Sie Chain-of-Custody-Prozeduren für die Beweishandhabung.
- Konfigurieren Sie Zeitsynchronisation (NTP) über alle Systeme, um sicherzustellen, dass Log-Zeitstempel über Komponenten hinweg korreliert sind. Ohne synchronisierte Zeitstempel wird die Rekonstruktion der Ereignisreihenfolge über mehrere Systeme hinweg unzuverlässig.
- Bewahren Sie Logs für einen für forensische Untersuchung ausreichenden Zeitraum auf — typischerweise 1-3 Jahre für Sicherheitslogs. Legacy-Systeme rotieren Logs oft aggressiv, um Speicherplatz zu sparen, was Beweise zerstört, bevor sie gebraucht werden.
- Etablieren Sie Beziehungen zu Rechts- und Strafverfolgungskontakten, bevor Vorfälle eintreten. Die Beweisanforderungen und Meldepflichten im Voraus zu kennen ermöglicht schnellere, effektivere Reaktion.
- Führen Sie Tabletop-Übungen durch, die Sicherheitsvorfälle simulieren und den forensischen Untersuchungsprozess durchgehen, um Lücken in Beweiserfassungs- und Analysefähigkeiten zu identifizieren.

## Tradeoffs ⇄

> Digitale-Forensik-Bereitschaft ermöglicht effektive Vorfalluntersuchung und rechtliche Schritte, erfordert aber proaktive Investition in Logging, Speicher und geschultes Personal.

**Vorteile:**

- Ermöglicht gründliche Untersuchung von Sicherheitsvorfällen, indem die Beweise bewahrt und organisiert werden, die nötig sind, um zu bestimmen, was geschah, wie und von wem.
- Unterstützt rechtliche Verfahren und regulatorische Meldung durch Bewahrung von Beweisen mit ordnungsgemäßer Chain of Custody.
- Verbessert die Vorfallreaktion, indem die Daten bereitgestellt werden, die nötig sind, um Umfang und Auswirkung von Verstößen schnell zu verstehen.
- Schreckt Insider-Bedrohungen ab, indem etabliert wird, dass forensische Untersuchungsfähigkeiten existieren und genutzt werden.

**Kosten und Risiken:**

- Umfassendes Logging und lange Aufbewahrungsfristen erfordern erhebliche Speicherkapazität, besonders für hochvolumige Legacy-Systeme.
- Forensische Untersuchung erfordert spezialisierte Fähigkeiten, die im bestehenden Team möglicherweise nicht existieren, was Schulung oder externe Expertise erfordert.
- Detailliertes Logging zu forensischen Zwecken kann unbeabsichtigt sensible Daten erfassen, die ihren eigenen Schutz benötigen.
- Beweiserfassungsprozeduren müssen sorgfältig befolgt werden, um rechtliche Zulässigkeit zu bewahren, und Fehler können Beweise ungültig machen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Digitale-Forensik-Bereitschaft effektive Vorfalluntersuchung in Legacy-Systemen ermöglicht.

Ein Legacy-Finanzsystem erkennt ungewöhnliche Transaktionsmuster, die auf unautorisierten Zugriff hindeuten. Das Sicherheitsteam versucht zu untersuchen, entdeckt aber, dass Anwendungslogs täglich rotiert und nur 7 Tage aufbewahrt werden — die verdächtige Aktivität erstreckt sich über drei Wochen, und die frühesten Beweise wurden permanent gelöscht. Nach diesem Vorfall implementiert das Team zentralisierten Log-Versand an einen dedizierten Forensik-Log-Speicher mit 2-jähriger Aufbewahrung, aktiviert Datenbank-Audit-Logging, das allen Datenzugriff und -änderungen erfasst, und fügt Netzwerk-Flow-Logging an der Netzwerksegmentgrenze des Legacy-Systems hinzu. Als sechs Monate später ein nachfolgender Vorfall eintritt, rekonstruiert das Forensik-Team eine vollständige Zeitlinie der Aktionen des Angreifers über einen Zeitraum von 45 Tagen, identifiziert den anfänglichen Zugriffsvektor (ein kompromittiertes Service-Konto), bestimmt genau, auf welche Datensätze zugegriffen wurde, und liefert die Beweise, die sowohl für regulatorische Meldung als auch für interne Disziplinarmaßnahmen nötig sind.
