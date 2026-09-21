---
title: Backup und Recovery
description: Sicherstellung regelmäßiger Datensicherung und Wiederherstellbarkeit.
category:
- Security
- Operations
problems:
- system-outages
- silent-data-corruption
- data-migration-integrity-issues
- missing-rollback-strategy
- deployment-risk
- regulatory-compliance-drift
- configuration-drift
layout: solution
lang: de
en_slug: backup-and-recovery
related_solutions:
- slug: regular-backups
  similarity: 0.9
- slug: restore-points
  similarity: 0.8
- slug: disaster-recovery
  similarity: 0.8
- slug: incident-response-measures
  similarity: 0.8
- slug: encryption
  similarity: 0.8
- slug: redundant-data-storage
  similarity: 0.75
---

## Description

Backup und Recovery ist die Disziplin, unabhängige, verifizierte Kopien der Daten und des Zustands eines Systems zu pflegen, sodass es nach Korruption, Ausfall oder Angriff auf einen bekannt guten Zeitpunkt zurückgesetzt werden kann, geregelt durch explizite Recovery-Point- und Recovery-Time-Ziele, die festlegen, wie viel Datenverlust und wie viel Ausfallzeit akzeptabel sind. Der Mechanismus funktioniert nur, wenn er getestet wird: Ein Backup, das nie wiederhergestellt wurde, ist funktional unverifiziert, und Legacy-Systeme sind besonders anfällig für genau diesen Fehlermodus, weil vor Jahren konfigurierte Backup-Jobs still laufen und selten überprüft werden, bis zu dem Moment, in dem sie benötigt werden und sich herausstellt, dass sie unbemerkt fehlgeschlagen sind. Legacy-Systeme verstärken das Risiko weiter, indem sie mehrere voneinander abhängige Datenspeicher anhäufen — Datenbanken, Dateisysteme, Konfiguration, Zertifikate, Verschlüsselungsschlüssel —, jeder möglicherweise gesichert (oder nicht) durch einen anderen, undokumentierten Prozess, eingerichtet von jemandem, der nicht mehr im Unternehmen ist. Echtes Backup und Recovery in diesem Kontext zu etablieren bedeutet, jeden Datenspeicher zu inventarisieren, eine konsistente Strategie wie die 3-2-1-Regel über alle hinweg anzuwenden und dann Wiederherstellbarkeit durch geplante Wiederherstellungsübungen zu beweisen, statt anzunehmen, dass ein fehlerfrei abgeschlossener Backup-Job bedeutet, dass die Daten tatsächlich wiederherstellbar sind. Dies ist die letzte Verteidigungslinie in einer Modernisierungsbemühung — es ist, was jede andere riskante Änderung (Migrationen, Schema-Evolution, Deployments) umkehrbar macht statt einer Einbahnstraße.

## How to Apply ◆

> Legacy-Systeme haben oft Backup-Prozeduren, die vor Jahren konfiguriert und nie auf tatsächliche Wiederherstellbarkeit getestet wurden. Backup und Recovery stellt sicher, dass Daten nach Ausfällen, Korruption oder Sicherheitsvorfällen verlässlich wiederhergestellt werden können.

- Inventarisieren Sie alle Datenspeicher im Legacy-System — Datenbanken, Dateisysteme, Konfigurationsdateien, Anwendungszustand, Zertifikate und Verschlüsselungsschlüssel — und verifizieren Sie, dass jeder eine angemessene Backup-Strategie hat.
- Implementieren Sie die 3-2-1-Backup-Regel: Pflegen Sie mindestens drei Kopien kritischer Daten, auf mindestens zwei verschiedenen Medientypen, mit mindestens einer Kopie extern oder in einer anderen Verfügbarkeitszone gespeichert.
- Definieren Sie Recovery-Point-Ziele (RPO) und Recovery-Time-Ziele (RTO) für jeden Datenspeicher basierend auf Geschäftsanforderungen. Das RPO definiert den maximal akzeptablen Datenverlust (wie oft Backups laufen), und das RTO definiert die maximal akzeptable Ausfallzeit (wie schnell Wiederherstellungen abgeschlossen sein müssen).
- Testen Sie die Backup-Wiederherstellung regelmäßig — mindestens vierteljährlich —, indem Sie tatsächliche Wiederherstellungsoperationen in eine separate Umgebung durchführen und die Datenintegrität verifizieren. Ein ungetestetes Backup ist kein Backup; es ist eine Hoffnung.
- Implementieren Sie automatisierte Backup-Verifikation, die die Integrität der Backup-Dateien prüft (Prüfsummen, Größenvalidierung), nachdem jedes Backup abgeschlossen ist. Viele Legacy-Backup-Fehler werden erst entdeckt, wenn eine Wiederherstellung versucht wird.
- Sichern Sie den Backup-Speicher mit Verschlüsselung im Ruhezustand, Zugriffskontrollen und Unveränderlichkeitsschutz, um zu verhindern, dass Ransomware Backup-Kopien verschlüsselt oder löscht.
- Dokumentieren Sie die Wiederherstellungsprozedur Schritt für Schritt, einschließlich der Reihenfolge der Operationen zur Wiederherstellung mehrerer voneinander abhängiger Systeme. Stellen Sie sicher, dass mehrere Teammitglieder die Prozedur ausführen können, nicht nur die Person, die die Backups eingerichtet hat.

## Tradeoffs ⇄

> Verlässliches Backup und Recovery bietet das ultimative Sicherheitsnetz gegen Datenverlust, erfordert aber laufende Investition in Speicher, Tests und Prozesspflege.

**Vorteile:**

- Bietet Wiederherstellbarkeit von Hardwareausfällen, Softwarefehlern, menschlichen Fehlern, Ransomware-Angriffen und Naturkatastrophen.
- Unterstützt Compliance-Anforderungen, die Datenwiederherstellbarkeit und Geschäftskontinuitätsplanung vorschreiben.
- Ermöglicht zuversichtliche Deployment- und Migrationsoperationen, im Wissen, dass Daten wiederhergestellt werden können, falls Änderungen Korruption verursachen.
- Verringert die Geschäftsauswirkung von Sicherheitsvorfällen, indem Wiederherstellung auf einen bekannt guten Zustand ermöglicht wird.

**Kosten und Risiken:**

- Backup-Speicherkosten wachsen mit Datenvolumen und Aufbewahrungsdauer, besonders für Legacy-Systeme mit großen, wachsenden Datenbanken.
- Backup-Fenster für große Legacy-Datenbanken können die Systemperformance während der Backup-Ausführung beeinträchtigen.
- Backups, die nicht regelmäßig getestet werden, könnten bei Bedarf aufgrund von Korruption, inkompatiblen Formaten oder unvollständiger Erfassung aller benötigten Daten fehlschlagen.
- Wiederherstellungsprozeduren für komplexe Legacy-Systeme mit mehreren voneinander abhängigen Datenspeichern sind ohne gründliche Dokumentation und Übung fehleranfällig und zeitaufwendig.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Backup- und Recovery-Praktiken Legacy-Systeme vor Datenverlust schützen.

Die Datenbank eines Legacy-ERP-Systems wird nach einem Storage-Controller-Ausfall korrumpiert. Das Betriebsteam versucht, aus dem neuesten Backup wiederherzustellen, und entdeckt, dass der Backup-Job seit drei Wochen still fehlgeschlagen ist, aufgrund einer vollen Backup-Festplatte, die niemand überwacht hat. Das Unternehmen verliert drei Wochen Transaktionsdaten. Nach diesem Vorfall implementiert das Team automatisiertes Backup-Monitoring, das verifiziert, dass jedes Backup erfolgreich abgeschlossen wird, die Backup-Dateigröße gegen erwartete Bereiche prüft und sofort bei jedem Fehler alarmiert. Sie führen außerdem monatliche Wiederherstellungstests ein, bei denen die vollständige Datenbank in eine Testumgebung wiederhergestellt wird und eine Reihe von Validierungsabfragen die Datenintegrität bestätigt. Als acht Monate später ein weiterer Festplattenausfall auftritt, wird die Wiederherstellung in 4 Stunden aus einem Backup abgeschlossen, das 6 Stunden zuvor erstellt wurde, deutlich innerhalb des definierten RPO von 24 Stunden und RTO von 8 Stunden.

Ein Legacy-Content-Management-System wird von einem Ransomware-Angriff getroffen, der die Anwendungsdatenbank und den gesamten angeschlossenen Dateispeicher verschlüsselt. Die bestehenden Backups werden auf einem Netzwerklaufwerk gespeichert, das die Ransomware ebenfalls verschlüsselt. Die Organisation verliert alle in den letzten 30 Tagen erstellten Inhalte. Nach der Wiederherstellung implementiert das Team unveränderlichen Backup-Speicher unter Nutzung von Write-Once-Cloud-Speicher, was sicherstellt, dass Backups für eine Mindestaufbewahrungsdauer von 90 Tagen nicht geändert oder gelöscht werden können. Sie fügen außerdem Air-Gapped-Backups hinzu, die physisch vom Netzwerk getrennt sind. Eine im folgenden Quartal durchgeführte Wiederherstellungsübung demonstriert, dass das vollständige System innerhalb des 12-Stunden-RTO aus unveränderlichen Backups wiederhergestellt werden kann, und die Backups überstehen ein simuliertes Ransomware-Szenario in der Testumgebung.
