---
title: Digitale Signaturen
description: Nutzung kryptografischer Signaturen für Code-Signierung, Dokumentenverifikation
  und Nachweis der Urheberschaft.
category:
- Security
problems:
- insecure-data-transmission
- data-protection-risk
- silent-data-corruption
- regulatory-compliance-drift
- secret-management-problems
- deployment-risk
layout: solution
lang: de
en_slug: digital-signatures
related_solutions:
- slug: authentication
  similarity: 0.75
- slug: key-management
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: cryptographic-methods
  similarity: 0.7
- slug: certificate-management
  similarity: 0.7
- slug: audit-trail-management
  similarity: 0.7
---

## Description

Digitale Signaturen nutzen asymmetrische Kryptografie, um ein Stück Inhalt — Code, ein Dokument, eine Datennachricht — an den privaten Schlüssel dessen zu binden, der es signiert hat, und erzeugen einen verifizierbaren Nachweis, dass der Inhalt seit der Signierung nicht verändert wurde und dass er echt von der behaupteten Partei stammt. Jeder Empfänger, der den entsprechenden öffentlichen Schlüssel besitzt, kann diesen Nachweis unabhängig prüfen, ohne dem Übertragungskanal selbst vertrauen zu müssen, was Signaturen grundlegend von perimeterbasierten Vertrauensmodellen unterscheidet, auf die sich Legacy-Systeme tendenziell verlassen. Viele Legacy-Systeme wurden gebaut, als Netzwerkgrenzen als vertrauenswürdig angenommen wurden und interne Dateifreigaben, Deployment-Pipelines und Partnerdatenaustausch keinerlei Integritätsverifikation trugen, was sie strukturell unfähig macht, Manipulation zu erkennen, selbst wenn sie offen sichtbar geschieht. Digitale Signaturen in ein solches System einzuführen schließt diese Lücke für die wichtigsten Artefakte: Build-Ausgaben, die in eine Deployment-Pipeline eintreten, Datenbankmigrationen und mit externen Partnern ausgetauschte Nachrichten können alle unabhängig verifiziert werden, bevor ihnen vertraut oder auf ihrer Basis gehandelt wird. Dies ist besonders für Modernisierungsbemühungen relevant, da es genau die alternden, informell gesicherten Übertragungs- und Deployment-Pfade der Legacy-Infrastruktur sind, denen heute am wahrscheinlichsten jede Integritätsprüfung fehlt. Die Technik beweist Echtheit und Integrität, nicht Korrektheit oder Sicherheit, sodass sie mit anderen Kontrollen kombiniert werden muss, um bösartigen Inhalt abzufangen, der zufällig ordnungsgemäß signiert ist.

## How to Apply ◆

> Legacy-Systemen fehlen oft Mechanismen, um die Echtheit und Integrität von Code, Dokumenten und Daten zu verifizieren — was es unmöglich macht, Manipulation zu erkennen oder Urheberschaft zu beweisen. Digitale Signaturen liefern kryptografischen Beweis, dass Inhalt seit der Signierung durch eine bekannte Partei nicht verändert wurde.

- Implementieren Sie Code-Signierung für alle deploybaren Artefakte (Binärdateien, Pakete, Container-Images, Skripte), sodass die Deployment-Pipeline verifizieren kann, dass nur autorisierter, unmodifizierter Code Produktion erreicht.
- Signieren Sie Datenbankmigrationen und Konfigurationsänderungen, sodass angewandte Änderungen gegen ihre signierten Originale verifiziert werden können, was unautorisierte Modifikationen erkennt.
- Nutzen Sie digitale Signaturen für systemübergreifenden Datenaustausch, um sicherzustellen, dass von externen Partnern empfangene Nachrichten während der Übertragung nicht manipuliert wurden und vom behaupteten Absender stammen.
- Implementieren Sie Dokumentensignierung für audit-kritische Dokumente, Verträge und vom Legacy-System erzeugte Berichte, um Nichtabstreitbarkeit und Manipulationsnachweis zu bieten.
- Etablieren Sie einen Schlüsselverwaltungsprozess für Signaturschlüssel: Erzeugen Sie Schlüssel mittels Hardware-Sicherheitsmodulen, verteilen Sie öffentliche Schlüssel über vertrauenswürdige Kanäle, implementieren Sie Schlüsselrotationszeitpläne und definieren Sie Prozeduren für die Reaktion auf Schlüsselkompromittierung.
- Verifizieren Sie Signaturen am Konsumptionspunkt, nicht nur am Erstellungspunkt. Jede Komponente, die signierten Inhalt empfängt, sollte die Signatur unabhängig verifizieren, bevor sie ihn verarbeitet.

## Tradeoffs ⇄

> Digitale Signaturen bieten Manipulationserkennung und Urheberschaftsnachweis, erfordern aber PKI-Infrastruktur und Schlüsselverwaltungsdisziplin.

**Vorteile:**

- Erkennt unautorisierte Modifikation von Code, Daten und Dokumenten durch kryptografische Integritätsverifikation.
- Bietet Nichtabstreitbarkeit — der Unterzeichner kann nicht abstreiten, den Inhalt signiert zu haben, was für rechtliche und Compliance-Zwecke wichtig ist.
- Verhindert Lieferkettenangriffe, indem sichergestellt wird, dass deployter Code derselbe Code ist, der gebaut und genehmigt wurde.
- Ermöglicht automatisierte Vertrauensentscheidungen, bei denen Systeme die Echtheit empfangener Daten ohne menschliches Eingreifen verifizieren können.

**Kosten und Risiken:**

- Schlüsselverwaltung fügt operative Komplexität hinzu — kompromittierte Signaturschlüssel können genutzt werden, um bösartigen Inhalt zu signieren, und verlorene Schlüssel verhindern die Verifikation zuvor signierten Inhalts.
- Signaturverifikation fügt jeder Operation, die signierten Inhalt betrifft, Verarbeitungszeit hinzu.
- Der Übergang von Legacy-Systemen zu signierten Artefakten erfordert Tooling-Änderungen in Build-Pipelines, Deployment-Prozessen und Datenaustauschschnittstellen.
- Digitale Signaturen beweisen Integrität und Echtheit, aber nicht Korrektheit — signierte Malware ist immer noch Malware.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie digitale Signaturen die Integrität von Legacy-Systemen schützen.

Ein Legacy-Deployment-Prozess kopiert kompilierte Java-WAR-Dateien von einem gemeinsamen Netzlaufwerk auf Produktions-Anwendungsserver. Ein Angreifer, der Zugriff auf das Netzlaufwerk erhält, ersetzt eine legitime WAR-Datei durch eine modifizierte Version mit einer Hintertür. Die Modifikation bleibt unentdeckt, weil es keine Integritätsverifikation im Deployment-Prozess gibt. Das Team implementiert Code-Signierung: Die CI/CD-Pipeline signiert jedes Build-Artefakt mit einem in einem Hardware-Sicherheitsmodul gespeicherten GPG-Schlüssel, und das Deployment-Skript verifiziert die Signatur vor dem Deployment. Als der Angreifer erneut eine WAR-Datei auf dem Netzlaufwerk modifiziert, erkennt das Deployment-Skript die ungültige Signatur und verweigert das Deployment, was das Sicherheitsteam über den Kompromittierungsversuch alarmiert.

Ein Legacy-EDI-System (Electronic Data Interchange) tauscht Bestellungen und Rechnungen mit Lieferanten über ungeschützte Dateiübertragungen aus. Ein Man-in-the-Middle-Angriff modifiziert Rechnungsbeträge während der Übertragung und leitet Zahlungen auf das Bankkonto des Angreifers um. Das Team implementiert digitale Signaturen für alle EDI-Dokumente: Der Absender signiert jedes Dokument mit seinem privaten Schlüssel, und der Empfänger verifiziert die Signatur vor der Verarbeitung. Als ein nachfolgender Modifikationsversuch auftritt, schlägt die Signaturverifikation fehl, und die modifizierte Rechnung wird abgelehnt. Der Absender wird benachrichtigt, erneut zu übertragen, und das Sicherheitsteam untersucht den Netzwerkpfad, um den Abfangpunkt zu identifizieren.
