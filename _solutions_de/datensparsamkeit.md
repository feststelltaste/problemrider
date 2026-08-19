---
title: Datensparsamkeit
description: Nur personenbezogene Daten erheben und speichern, die für den Zweck
  notwendig sind.
category:
- Security
- Architecture
problems:
- data-protection-risk
- regulatory-compliance-drift
- unbounded-data-growth
- high-database-resource-utilization
- silent-data-corruption
- insufficient-audit-logging
- slow-database-queries
- inadequate-test-data-management
- retention-obligations-block-change
layout: solution
lang: de
en_slug: datensparsamkeit
related_solutions:
- slug: data-flow-control
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: least-privilege
  similarity: 0.7
- slug: retention-and-disposal-policy
  similarity: 0.7
- slug: backup-and-recovery
  similarity: 0.65
- slug: privacy-by-design
  similarity: 0.65
---

## Description

Datensparsamkeit beschränkt, welche personenbezogenen und sensiblen Daten ein System erhebt und wie lange es sie behält, auf nur das, was für einen aktiven, spezifischen Geschäftszweck strikt notwendig ist, und ersetzt die Standard-Legacy-Haltung, alles auf unbestimmte Zeit zu sammeln und zu behalten, nur für den Fall. Sie anzuwenden bedeutet, jedes gespeicherte Datenelement zu prüfen, um festzustellen, ob der Geschäftszweck, dem es einst diente, noch gültig ist, die Erhebung von Feldern an der Quelle zu entfernen, sobald dieser Zweck nicht mehr gilt, und automatisierte Aufbewahrungs- und Lösch- oder Anonymisierungsprozesse zu implementieren, statt sich auf manuelle Bereinigung zu verlassen, die dazu neigt, auf unbestimmte Zeit aufgeschoben zu werden. Dies ist für Legacy-Systeme besonders folgenreich, weil sie typischerweise gebaut und erweitert wurden, lange bevor Datenschutzvorschriften Datensparsamkeit zu einer rechtlichen Erwartung machten, und infolgedessen neigen sie dazu, still Jahre personenbezogener Daten für nicht mehr aktive Kunden und Felder anzuhäufen, die einst benötigt wurden, aber von keinem aktuellen Prozess mehr genutzt werden. Diesen Fußabdruck zu reduzieren verringert direkt den Schaden, den ein zukünftiger Datenschutzverstoß oder unautorisierter Zugriffsvorfall verursachen kann — Daten, die nie behalten wurden, können nicht exfiltriert werden —, und es verengt gleichzeitig den Umfang der Compliance-Verpflichtungen und der Sicherheitskontrollen, die nötig sind, um sie zu erfüllen. Das entsprechende Risiko ist, dass historische Daten, einmal gelöscht oder anonymisiert, nicht wiederhergestellt werden können, falls ein legitimer zukünftiger Geschäftsbedarf dafür entsteht, und dass Datenabhängigkeiten über vernetzte Legacy-Systeme hinweg selten gut genug dokumentiert sind, um zu garantieren, dass das Entfernen von Daten in einem System nicht still etwas in einem anderen bricht.

## How to Apply ◆

> Legacy-Systeme neigen dazu, alle verfügbaren Daten unbegrenzt zu erheben und zu behalten, oft einschließlich sensibler Informationen, die für keinen Geschäftszweck mehr benötigt werden. Datensparsamkeit reduziert Risiko, indem Datenerhebung und -aufbewahrung auf das strikt Notwendige begrenzt werden.

- Prüfen Sie alle im Legacy-System gespeicherten personenbezogenen und sensiblen Daten. Bestimmen Sie für jedes Datenelement den spezifischen Geschäftszweck, dem es dient, und ob dieser Zweck noch gültig ist. Daten, die „nur für den Fall" oder „weil wir es schon immer haben" erhoben wurden, sollten eliminiert werden.
- Implementieren Sie Datenaufbewahrungsrichtlinien mit spezifischen Ablauffristen für jede Datenkategorie. Personenbezogene Daten sollten automatisch gelöscht oder anonymisiert werden, wenn sie für ihren angegebenen Zweck nicht mehr benötigt werden.
- Entfernen Sie unnötige Datenerhebung aus Eingabeformularen und APIs. Legacy-Systeme erheben oft Felder, die einst erforderlich waren, aber nicht mehr genutzt werden — diese Felder an der Quelle zu entfernen verhindert, dass unnötige Daten ins System gelangen.
- Anonymisieren oder pseudonymisieren Sie Daten, die für Analytics, Testing und Entwicklungsumgebungen genutzt werden. Vollständige Produktionsdaten mit echten personenbezogenen Informationen sollten nie in Nicht-Produktionskontexten genutzt werden.
- Implementieren Sie automatisierte Datenbereinigungsprozesse, die abgelaufene Daten gemäß Aufbewahrungsrichtlinien löschen. Manuelle Bereinigungsprozesse sind unzuverlässig und werden tendenziell auf unbestimmte Zeit aufgeschoben.
- Überprüfen Sie Vereinbarungen zur Weitergabe von Daten an Dritte, um sicherzustellen, dass nur notwendige Daten mit externen Partnern geteilt werden und dass geteilte Daten denselben Sparsamkeits- und Aufbewahrungsstandards unterliegen.
- Dokumentieren Sie die Rechtsgrundlage und geschäftliche Rechtfertigung für jede Kategorie personenbezogener Daten, die das System erhebt und aufbewahrt. Diese Dokumentation ist von der DSGVO gefordert und dient als Grundlage für Sparsamkeitsentscheidungen.

## Tradeoffs ⇄

> Datensparsamkeit reduziert Risiko und Kosten von Datenschutzverstößen, vereinfacht Compliance und reduziert Speicherkosten, erfordert aber sorgfältige Analyse von Datenabhängigkeiten und kann künftige Analytics-Fähigkeiten einschränken.

**Vorteile:**

- Reduziert die Auswirkung von Datenschutzverstößen, indem die Menge sensibler Daten begrenzt wird, die exfiltriert werden können — Sie können keine Daten verlieren, die Sie nicht haben.
- Vereinfacht Compliance mit Datenschutzvorschriften (DSGVO, CCPA), die Datensparsamkeit als Kernprinzip vorschreiben.
- Reduziert Speicherkosten und Datenbankkomplexität, indem unnötige Datenanhäufung eliminiert wird.
- Verringert Umfang und Kosten von Sicherheitskontrollen, indem das Volumen der zu schützenden Daten reduziert wird.

**Kosten und Risiken:**

- Gelöschte historische Daten können nicht wiederhergestellt werden, falls ein zukünftiger Geschäftsbedarf identifiziert wird, was sorgfältige Analyse vor der Löschung erfordert.
- Datenabhängigkeiten über vernetzte Legacy-Systeme hinweg sind möglicherweise nicht vollständig dokumentiert, und das Löschen von Daten in einem System kann Funktionalität in einem anderen brechen.
- Die Implementierung von Datenaufbewahrungsrichtlinien in Legacy-Datenbanken ohne zeitliche Metadaten erfordert das Hinzufügen von Infrastruktur zur Ablaufverfolgung.
- Stakeholder könnten sich gegen Datensparsamkeit sträuben, aus Sorge, künftige Analytics- oder Berichtsfähigkeiten zu verlieren.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Datensparsamkeit Risiko in Legacy-Systemen reduziert.

Ein Legacy-Kundenbeziehungsmanagementsystem hat 15 Jahre Kundendaten angehäuft, einschließlich Heimadressen, Telefonnummern, Geburtsdaten und Kaufhistorien für Kunden, die seit über 10 Jahren keinen Kauf getätigt haben. Eine DSGVO-Auskunftsanfrage zeigt, dass das System personenbezogene Daten für 3,2 Millionen Kunden speichert, von denen nur 400.000 aktiv sind. Das Team implementiert eine Datenaufbewahrungsrichtlinie: Inaktive Kundendatensätze werden nach 3 Jahren anonymisiert (Kaufhistorie wird ohne personenbezogene Identifikatoren für Geschäftsanalytics behalten, personenbezogene Daten werden gelöscht). Aktive Kundendatensätze werden jährlich überprüft, um Felder zu entfernen, die für aktuelle Geschäftsprozesse nicht mehr benötigt werden. Der personenbezogene Datenfußabdruck schrumpft um 75 Prozent, und als ein späterer Sicherheitsvorfall Datenbankdatensätze offenlegt, ist die Auswirkungsbewertung dramatisch kleiner, weil die offengelegten Datensätze für die Mehrheit der Einträge anonymisierte Daten enthalten.

Ein Legacy-Gesundheitsterminplanungssystem erhebt und speichert dauerhaft Versicherungspolicennummern, Sozialversicherungsnummern und vollständige Krankengeschichten von Patienten — obwohl das Terminplanungssystem nur Name, Geburtsdatum und Versicherungsverifikationsstatus braucht, um zu funktionieren. Das Team arbeitet mit den klinischen und Abrechnungsteams zusammen, um den minimalen Datensatz zu identifizieren, der für die Terminplanung benötigt wird, und entfernt alle unnötigen Felder aus dem System. Versicherungspolicennummern werden zum Buchungszeitpunkt gegen die API des Versicherungsanbieters verifiziert, aber nicht gespeichert. Sozialversicherungsnummern werden vollständig entfernt, da sie bereits im separaten klinischen Aktensystem gepflegt werden. Das vereinfachte Datenmodell reduziert den PCI- und HIPAA-Compliance-Umfang des Systems und eliminiert eine ganze Kategorie von Datenschutzverstoßrisiko.
