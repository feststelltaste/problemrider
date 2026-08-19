---
title: Datenflusskontrolle
description: Steuerung und Filterung von Datenflüssen zwischen Komponenten und Systemen.
category:
- Security
- Architecture
problems:
- insecure-data-transmission
- data-protection-risk
- authorization-flaws
- cross-system-data-synchronization-problems
- poor-interfaces-between-applications
- cascade-failures
- error-message-information-disclosure
layout: solution
lang: de
en_slug: data-flow-control
related_solutions:
- slug: encryption
  similarity: 0.75
- slug: authorization
  similarity: 0.75
- slug: logging-and-monitoring
  similarity: 0.75
- slug: input-validation
  similarity: 0.7
- slug: api-security
  similarity: 0.7
- slug: authentication
  similarity: 0.7
---

## Description

Datenflusskontrolle etabliert explizite Regeln dafür, welche Daten sich zwischen welchen Komponenten bewegen dürfen, in welcher Form und nach welcher Filterung oder Maskierung, statt Daten frei über jede Grenze fließen zu lassen, die ein Legacy-System zufällig hat. Sie erfordert zunächst, abzubilden, wie sich Daten tatsächlich durch das System bewegen — wo sie entstehen, was sie durchlaufen und wo Vertrauensgrenzen überschritten werden —, und dann an jeder dieser Grenzen durchzusetzen, dass nur die Felder übertragen werden, die ein Konsument tatsächlich braucht, dass sensible Werte maskiert oder tokenisiert werden, bevor sie Komponenten erreichen, die den echten Wert nicht brauchen, und dass in das System eintretende oder es verlassende Daten validiert und bereinigt werden. Dies ist in Legacy-Systemen akut wichtig, weil Komponenten typischerweise verdrahtet wurden, lange bevor jemand an Least-Privilege-Datenzugriff dachte: APIs geben jedem Konsumenten unabhängig vom Bedarf ganze Datenbankdatensätze zurück, Logging erfasst vollständige Request-Bodies einschließlich sensibler Felder, und implizites Vertrauen in den Netzwerkperimeter ersetzt jede Prüfung dessen, welche spezifischen Daten eine bestimmte Komponente sehen kann. Die Etablierung von Datenflusskontrolle verkleinert den Auswirkungsradius jeder einzelnen kompromittierten Komponente oder geleakten Logdatei, weil keine Komponente mehr Daten hält, als ihre Funktion erfordert, und sie gibt einer Organisation die Sichtbarkeit über ihre eigenen Datenbewegungen, die Vorschriften wie DSGVO oder HIPAA von ihr erwarten nachweisen zu können. Der Tradeoff ist, dass Legacy-Integrationsmuster — gemeinsam genutzte Datenbanken, Flat-File-Austausch, geschwätzige APIs — sich dagegen sträuben, an definierten Grenzen sauber getrennt zu werden, sodass die Einführung von Flusskontrolle ebenso sehr eine Übung in der Neuarchitektur von Schnittstellen ist wie im Hinzufügen von Filterlogik.

## How to Apply ◆

> Legacy-Systeme lassen Daten oft frei zwischen Komponenten passieren, ohne Filterung, Validierung oder Zugriffskontrolle, was Gelegenheiten für Datenlecks, Injection-Angriffe und unautorisierten Zugriff auf sensible Informationen schafft. Datenflusskontrolle etabliert explizite Regeln dafür, welche Daten sich zwischen welchen Komponenten bewegen können und in welcher Form.

- Bilden Sie alle Datenflüsse im Legacy-System ab: Identifizieren Sie, wo Daten entstehen, welche Transformationen sie durchlaufen, welche Komponenten sie passieren und wo sie gespeichert werden. Achten Sie besonders auf Flüsse, die Vertrauensgrenzen überschreiten (intern zu extern, Anwendung zu Datenbank, nutzerseitig zu Backend).
- Klassifizieren Sie Daten nach Sensibilitätsstufe und definieren Sie Handhabungsregeln für jede Klassifikation. Sensible Daten (PII, Finanzunterlagen, Gesundheitsinformationen) sollten identifiziert und verfolgt werden, während sie durch das System fließen, um angemessenen Schutz auf jeder Stufe sicherzustellen.
- Implementieren Sie Datenfilterung an Komponentengrenzen, um Felder zu entfernen, die die empfangende Komponente nicht braucht. Legacy-APIs geben oft ganze Datenbankdatensätze zurück, wenn der Konsument nur wenige Felder braucht, was sensible Daten unnötig exponiert.
- Fügen Sie an jeder überschrittenen Vertrauensgrenze Datenvalidierung und -bereinigung hinzu. Daten, die von einer nicht vertrauenswürdigen Quelle eintreten, müssen validiert werden, bevor sie verarbeitet werden, und Daten, die zu einem nicht vertrauenswürdigen Ziel austreten, müssen bereinigt werden, um Informationslecks zu verhindern.
- Implementieren Sie Datenmaskierung oder Tokenisierung für sensible Felder, die durch Zwischenkomponenten laufen, die die tatsächlichen Werte nicht brauchen. Zum Beispiel sollte ein Logging-System maskierte Kreditkartennummern erhalten, nicht vollständige Nummern.
- Nutzen Sie Kontrollen auf Netzwerkebene (Firewalls, Netzwerkrichtlinien, Service Mesh), um erlaubte Datenflusspfade durchzusetzen und unautorisierte direkte Verbindungen zwischen Komponenten zu verhindern, die nur über definierte Schnittstellen kommunizieren sollten.
- Prüfen Sie Datenflüsse periodisch, um sicherzustellen, dass sie der dokumentierten Flusskarte entsprechen und dass keine unautorisierten Datenpfade durch Konfigurationsänderungen oder Workarounds entstanden sind.

## Tradeoffs ⇄

> Datenflusskontrolle minimiert Datenexposition und setzt das Prinzip der geringsten Rechte auf Datenebene durch, erfordert aber umfassende Flussabbildung und laufende Governance.

**Vorteile:**

- Reduziert den Auswirkungsradius von Datenschutzverletzungen, indem sichergestellt wird, dass jede Komponente nur Zugriff auf die Daten hat, die sie braucht, was begrenzt, was von einem einzelnen Punkt aus exfiltriert werden kann.
- Verhindert, dass sensible Daten in Logs, Fehlermeldungen, Caches und anderen Orten lecken, an denen sie nicht erscheinen sollten.
- Unterstützt Compliance mit Datenschutzvorschriften (DSGVO, HIPAA), die nachweisbare Kontrolle darüber verlangen, wie personenbezogene Daten fließen und verarbeitet werden.
- Macht die Datenarchitektur des Systems sichtbar und auditierbar, was informierte Sicherheitsentscheidungen ermöglicht.

**Kosten und Risiken:**

- Alle Datenflüsse in einem komplexen Legacy-System abzubilden ist zeitaufwendig, und die resultierende Karte erfordert laufende Pflege, während sich das System weiterentwickelt.
- Übermäßig restriktive Datenflusskontrollen können bestehende Funktionalität brechen, die von Zugriff auf Daten abhängt, die nach der Filterung nicht mehr verfügbar sind.
- Datenmaskierung und Tokenisierung fügen Verarbeitungsoverhead und Komplexität hinzu, besonders wenn nachgelagerte Komponenten gelegentlich die Originalwerte brauchen.
- Legacy-Integrationsmuster (gemeinsam genutzte Datenbanken, Flat-File-Austausch) machen es schwierig, Datenflusskontrollen an Komponentengrenzen durchzusetzen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Datenflusskontrolle Datenexposition in Legacy-Systemen verhindert.

Ein Legacy-E-Commerce-System protokolliert zu Debugging-Zwecken den vollständigen HTTP-Request-Body, einschließlich Kundenkreditkartennummern und CVV-Codes in Zahlungsanfragen. Ein Sicherheitsaudit zeigt, dass drei Jahre Kreditkartendaten im Klartext in Anwendungslogdateien gespeichert sind, auf die 40 Entwickler zugreifen können. Das Team implementiert Datenflusskontrollen, indem eine Logging-Middleware hinzugefügt wird, die sensible Felder maskiert (Kreditkartennummern zeigen nur die letzten vier Ziffern, CVV-Codes werden durch Sternchen ersetzt), bevor sie das Logging-System erreichen. Sie implementieren auch eine Datenflussrichtlinie, die PII verbietet, ohne Maskierung in irgendein Logging-, Caching- oder Analytics-System zu fließen. Die bestehenden Logdateien mit unmaskierten Kreditkartendaten werden sicher gelöscht, und automatisiertes Scannen wird hinzugefügt, um zukünftige Fälle sensibler Daten in Logs zu erkennen.

Ein Legacy-HR-System bietet eine API, die sowohl vom internen Verzeichnis des Unternehmens als auch von einem Drittanbieter für Sozialleistungen konsumiert wird. Die API gibt beiden Konsumenten denselben vollständigen Mitarbeiterdatensatz zurück, einschließlich Gehaltsinformationen, Sozialversicherungsnummern und Leistungsbeurteilungsdaten. Der Sozialleistungsanbieter braucht nur Name, Geburtsdatum und Leistungsanmeldestatus. Das Team implementiert Datenflusskontrolle, indem konsumentenspezifische API-Views erstellt werden: Das interne Verzeichnis erhält eine gefilterte Antwort mit nur Name, Abteilung und Kontaktinformationen, während der Sozialleistungsanbieter nur die für die Leistungsverwaltung erforderlichen Felder erhält. Sozialversicherungsnummern werden während der Übertragung tokenisiert und können nur von autorisierten Komponenten aufgelöst werden. Dies reduziert die Exposition sensibler Daten von 45 Feldern auf 3-5 Felder pro Konsument.
