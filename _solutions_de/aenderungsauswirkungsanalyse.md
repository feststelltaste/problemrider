---
title: Änderungsauswirkungsanalyse
description: Ermittlung, was eine vorgeschlagene Änderung tatsächlich betrifft —
  Aufrufer, Daten, Konsumenten, Betrieb — bevor man sich darauf festlegt, mithilfe
  von Tooling statt Erinnerung.
category:
- Architecture
- Code
- Process
problems:
- hidden-dependencies
- hidden-side-effects
- rapid-system-changes
- large-estimates-for-small-changes
- fear-of-breaking-changes
- regression-bugs
- ripple-effect-of-changes
- high-defect-rate-in-production
- change-management-chaos
- circular-dependency-problems
- shared-dependencies
- tangled-cross-cutting-concerns
- increased-bug-count
- no-formal-change-control-process
- schema-evolution-paralysis
- shared-database
- approval-dependencies
- communication-risk-outside-project
- increasing-brittleness
- partial-bug-fixes
- entity-attribute-value-overuse
- core-modification-of-standard-software
layout: solution
lang: de
en_slug: change-impact-analysis
related_solutions:
- slug: mikado-method
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.75
- slug: code-hotspot-analysis
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
- slug: change-management-process
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
---

## Description

Änderungsauswirkungsanalyse ist die Praxis festzustellen, bevor eine Änderung vorgenommen wird, was sie sonst noch beeinflusst: welcher Code sie aufruft, welche Daten sie schreibt, welche nachgelagerten Systeme diese Daten lesen, welche Berichte davon abhängen, und was operative Prozeduren darüber annehmen. In einem gut verstandenen System geschieht dies implizit im Kopf desjenigen, der die Änderung vornimmt. In einem Legacy-System kann es das nicht, weil kein Einzelner ein vollständiges Bild hat und das Bild nirgendwo aufgeschrieben ist. Das Ergebnis ist der charakteristische Legacy-Fehler: eine Änderung, von der jeder zustimmte, dass sie klein sei, bricht etwas in einem Subsystem, mit dem niemand sie verbunden hatte. Auswirkungsanalyse ersetzt Erinnerung durch Evidenz, unter Nutzung der Artefakte, die tatsächlich existieren — der Code, das Schema, die Logs, die Versionskontrollhistorie —, um zu rekonstruieren, was Erinnerung nicht liefern kann.

## How to Apply ◆

> Die Abhängigkeiten, die in einem Legacy-System Probleme verursachen, sind selten die im Code sichtbaren; sie laufen durch die Datenbank, durch Datei-Drops, durch geplante Jobs und durch einen Bericht, den jemand im Finanzwesen monatlich ausführt.

- Beginnen Sie mit **statischer Analyse der Aufrufer**: wer ruft diesen Code auf, direkt und transitiv. Modernes Tooling handhabt dies gut innerhalb einer einzelnen Codebasis und ist der günstige erste Durchgang. Beachten Sie explizit, wo es aufhört zu funktionieren — Reflection, dynamischer Dispatch, konfigurationsgetriebener Aufruf und Stored Procedures sind für es alle unsichtbar.
- **Folgen Sie den Daten, nicht nur dem Code.** Identifizieren Sie, welche Tabellen die Änderung schreibt, und suchen Sie dann nach jedem Leser dieser Tabellen, einschließlich Berichtswerkzeugen, Batch-Jobs und anderen Anwendungen mit eigenen Datenbank-Anmeldeinformationen. In Systemen mit gemeinsam genutzter Datenbank liegt hier üblicherweise die echte Auswirkung, und dort findet statische Analyse nichts.
- Nutzen Sie **Laufzeit-Evidenz**, um zu erfassen, was statische Analyse übersieht: Produktionsprotokolle, Datenbank-Audit-Trails und Zugriffs-Telemetrie zeigen, wer tatsächlich eine Schnittstelle aufruft und eine Tabelle liest, einschließlich Konsumenten, die niemand dokumentiert hat. Eine Woche an Abfrageprotokollen identifiziert häufig Konsumenten, die keine Menge an Code-Lektüre gefunden hätte.
- Konsultieren Sie die **Versionskontrollhistorie** für zeitliche Kopplung — was sich historisch zusammen mit dem Code geändert hat, den Sie ändern wollen. Dateien, die wiederholt in denselben Commits auftauchen, sind auf eine Weise gekoppelt, die keine statische Analyse erkennt, und die Historie ist ein Protokoll dessen, was vergangene Entwickler auf die harte Tour entdeckt haben.
- Prüfen Sie die **operative Fläche**: Monitoring-Schwellen, Runbooks, geplante Jobs und Alarmdefinitionen, die auf das geänderte Verhalten verweisen. Eine Änderung, die im Code korrekt ist und still einen Alarm invalidiert, ist eine Änderung, die ein Sicherheitsnetz entfernt.
- Fragen Sie, **wer sonst ein Interesse hat**, für die Auswirkungen, die kein Werkzeug findet: die Integration eines externen Partners, ein regulatorischer Bericht, ein manueller Abgleich, den jemand monatlich durchführt. Verteilen Sie die spezifische Liste dessen, was sich ändert, statt einer allgemeinen Ankündigung, weil eine allgemeine Ankündigung keine Antwort erhält.
- **Erfassen Sie die Analyse mit der Änderung**, im Pull Request oder im Ticket. Der Befund — dass diese sieben Konsumenten existieren — ist teuer zu produzieren und wird von der nächsten Person, die den Bereich berührt, wieder benötigt.
- Nutzen Sie das Ergebnis, um **den Ansatz zu entscheiden, nicht nur um fortzufahren**. Eine Analyse, die elf Konsumenten findet, könnte dafür sprechen, das alte Verhalten hinter einer Schnittstelle beizubehalten, statt es zu ändern, was eine Designentscheidung ist, die die Analyse ermöglicht.
- **Begrenzen Sie den Aufwand explizit.** Auswirkungsanalyse kann sich in einem verworrenen System unbegrenzt ausdehnen. Setzen Sie einen Anteil des erwarteten Änderungsaufwands und stoppen Sie dort, wobei Sie festhalten, was nicht geprüft wurde, sodass das Restrisiko formuliert statt weggenommen wird.

## Tradeoffs ⇄

> Analyse vor der Änderung verhindert die teuren Überraschungen, auf Kosten der Zeit, die für Änderungen aufgewendet wird, die in Ordnung gewesen wären, und sie kann nie vollständig sein.

**Vorteile:**

- Unbekannte Konsumenten werden gefunden, bevor sie brechen statt danach, was der Unterschied zwischen einer Designentscheidung und einem Produktionsvorfall ist.
- Schätzungen werden für Legacy-Arbeit erheblich genauer, da der dominante Schätzfehler unbekannter Umfang statt falsch eingeschätzter Aufwand ist.
- Angst vor Veränderung sinkt mit Evidenz. Entwickler vermeiden es, Code zu berühren, weil sie die Konsequenzen nicht begrenzen können, und die Konsequenzen zu begrenzen ist genau das, was dies produziert.
- Die Analyse häuft sich an. Aufgezeichnete Befunde bauen schrittweise die Abhängigkeitskarte auf, die die Dokumentation des Systems nie enthielt.
- Sie informiert die Ansatzwahl: Den Explosionsradius früh zu kennen ist, was es einem Team erlaubt, einen additiven statt eines ändernden Pfads zu wählen.

**Kosten und Risiken:**

- Sie kostet Zeit bei jeder Änderung, einschließlich der Mehrheit, die harmlos gewesen wäre, und dieser Overhead wird sofort gespürt, während die vermiedenen Vorfälle unsichtbar sind.
- Vollständigkeit ist unerreichbar. Dynamisches Verhalten, Reflection und undokumentierte externe Konsumenten bedeuten, dass manche Auswirkung immer übersehen wird, und eine gründliche Analyse kann falsches Vertrauen erzeugen.
- Laufzeit-Evidenz deckt nur das Beobachtungsfenster ab. Ein vierteljährlicher Batch-Job wird nicht in einer Woche an Protokollen erscheinen, und diese seltenen Konsumenten sind oft die störendsten, wenn sie brechen.
- In einem stark verworrenen System kann die Analyse schlussfolgern, dass alles alles berührt, was akkurat und nicht umsetzbar ist, und Aufwand verbraucht, um das festzustellen.
- Für die Analyse aufgewendete Zeit ist manchmal besser darin investiert, die Änderung sicher rückgängig machbar zu gestalten, besonders wo ein schneller Rollback verfügbar und die Kosten eines kurzen Ausfalls gering sind.

## How It Could Be

Ein Entwickler wurde gebeten, das Format einer Kundenreferenznummer in einem Auftragsmanagementsystem zu ändern, geschätzt auf zwei Tage. Statische Analyse fand neun Aufrufstellen, alle unkompliziert. Den Daten statt dem Code zu folgen fand die Referenz in drei Tabellen gespeichert, von denen eine nachts von einem Data-Warehouse-Job einer anderen Abteilung gelesen wurde, und wöchentlich an einen Logistikpartner über eine Fixed-Width-Datei exportiert wurde, deren Spaltenbreiten in einem Dokument von 2008 definiert waren. Eine Abfrage des Datenbank-Audit-Logs über zehn Tage brachte einen vierten Leser ans Licht: ein Finanzberichtswerkzeug, das direktes SQL ausführte. Die Zweitagesänderung wurde zu einer sechswöchigen koordinierten Anstrengung — was die tatsächliche Größe der Änderung die ganze Zeit war. Die Alternative, entdeckt durch das Brechen der Partnerintegration, war demselben Team zwei Jahre zuvor passiert und hatte neun Wochen einschließlich der Wiederherstellung gedauert.

Ein zweites Team nutzte Auswirkungsanalyse, um einen Ansatz zu wählen, statt einen zu dimensionieren. Die vorgeschlagene Änderung modifizierte, wie Kontostände berechnet wurden. Die Analyse fand vierzehn Konsumenten, vier davon außerhalb der Kontrolle des Teams und zwei davon explizit abhängig von einem Rundungsverhalten, das die Änderung verändern würde. Statt die Berechnung zu ändern, fügte das Team eine neue Berechnung neben der alten hinzu, verschob Konsumenten einzeln, während jeder verifiziert wurde, und löschte die ursprüngliche elf Monate später, als der letzte Konsument migriert hatte. Der Gesamtaufwand war größer als die direkte Änderung gewesen wäre, und es gab keinen Vorfall, keine Koordinationskrise und keine Partnereskalation.
