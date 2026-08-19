---
title: Autorisierungskonzept
description: Definition des Zugriffs auf kritische Daten und Funktionen.
category:
- Security
- Management
problems:
- authorization-flaws
- data-protection-risk
- regulatory-compliance-drift
- poorly-defined-responsibilities
- insufficient-audit-logging
- authentication-bypass-vulnerabilities
- authorization-role-explosion
layout: solution
lang: de
en_slug: authorization-concept
related_solutions:
- slug: authorization
  similarity: 0.85
- slug: domain-based-authorization-concept
  similarity: 0.85
- slug: role-based-access-control
  similarity: 0.75
- slug: least-privilege
  similarity: 0.75
- slug: role-model-rationalization
  similarity: 0.7
- slug: authentication
  similarity: 0.7
---

## Description

Ein Autorisierungskonzept ist das dokumentierte, menschenlesbare Design, das jeder technischen Zugriffskontrollimplementierung vorausgeht und sie steuert: Es definiert, welche Datenklassifizierungen existieren, welche Rollen Zugriff auf jede benötigen, welche Operationen jede Rolle durchführen darf, und die Aufgabentrennungs- und Überprüfungsprozesse, die diese Gewährungen über die Zeit ehrlich halten. Statt eines Durchsetzungsmechanismus selbst ist es der Bauplan, den Autorisierung und rollenbasierte Zugriffskontrolle implementieren — das Artefakt, das es jemandem erlaubt, „sollte diese Rolle diese Berechtigung haben" aus einer geschriebenen Begründung zu beantworten statt aus institutionellem Gedächtnis oder historischem Zufall. Legacy-Systeme hatten typischerweise nie ein solches Dokument: Berechtigungen wurden individuell, als Reaktion auf spezifische Anfragen, über viele Jahre gewährt, ohne dass jemand beauftragt war, periodisch zu prüfen, ob das resultierende Muster noch sinnvoll war — so gelangen Systeme in den Zustand, in dem große Anteile der Nutzer Zugriff besitzen, den sie nicht mehr benötigen. Der Aufbau eines Autorisierungskonzepts für ein bestehendes Legacy-System ist daher primär eine Entdeckungs- und Abgleichsarbeit — die Zugriffsmuster, die tatsächlich existieren, zurückzuentwickeln, sie mit dem zu vergleichen, was das Geschäft tatsächlich benötigt, und den Unterschied als explizites, überprüfbares Modell statt eines impliziten zu formalisieren. Ihr Wert zeigt sich direkt in Compliance-Audits, wo ein System mit geschriebenem Autorisierungskonzept seine Zugriffskontrollbegründung demonstrieren kann, während ein System ohne eines nur beschreiben kann, was beobachtet wurde.

## How to Apply ◆

> Legacy-Systemen fehlt oft ein dokumentiertes Autorisierungskonzept, was zu Ad-hoc-Berechtigungszuweisungen führt, die sich über Jahre anhäufen. Ein Autorisierungskonzept definiert ein klares Modell dafür, wer auf welche Daten und Funktionen zugreifen kann, und dient als Bauplan für die Implementierung.

- Dokumentieren Sie alle Datenklassifizierungen im Legacy-System (öffentlich, intern, vertraulich, eingeschränkt) und kartieren Sie, welche Nutzerrollen Zugriff auf welche Klassifizierungsebene benötigen. Dies schafft die Grundlage für ein auf Geschäftsanforderungen basierendes Berechtigungsmodell.
- Definieren Sie funktionale Berechtigungen, indem Geschäftsprozesse auf die Systemoperationen abgebildet werden, die sie benötigen. Listen Sie für jede Rolle die spezifischen Erstellungs-, Lese-, Änderungs- und Löschoperationen auf, die für jeden Ressourcentyp erlaubt sind.
- Etablieren Sie das Prinzip der geringsten Berechtigung als Designregel: Jede Rolle erhält nur die minimalen Berechtigungen, die zur Durchführung ihrer Geschäftsfunktion notwendig sind. Dokumentieren Sie die Begründung für jede Berechtigungsgewährung, sodass sie während Audits überprüft und hinterfragt werden kann.
- Erstellen Sie eine Rollenhierarchie, die die organisatorische Struktur widerspiegelt und Berechtigungsvererbung vermeidet, die breiteren Zugriff gewährt als beabsichtigt. Dokumentieren Sie, welche Berechtigungen vererbt und welche explizit zugewiesen werden.
- Definieren Sie Aufgabentrennungsregeln, die verhindern, dass ein einzelner Nutzer widersprüchliche Operationen durchführt (z. B. eine Finanztransaktion erstellen und genehmigen). Implementieren Sie diese als harte Beschränkungen im Autorisierungssystem.
- Etablieren Sie einen formalen Prozess zur Anfrage, Gewährung, Überprüfung und Widerrufung von Zugriff. Beziehen Sie verpflichtende Genehmigungsworkflows, zeitlich begrenzte Zugriffsgewährungen für temporäre Bedürfnisse und automatischen Widerruf bei Rollenwechsel ein.
- Planen Sie periodische Zugriffsüberprüfungen (vierteljährlich oder halbjährlich), bei denen Rolleneigentümer verifizieren, dass alle zugewiesenen Berechtigungen weiterhin notwendig und angemessen sind.

## Tradeoffs ⇄

> Ein gut definiertes Autorisierungskonzept bietet klare, auditierbare Zugriffskontrolle, die mit Geschäftsbedürfnissen übereinstimmt, erfordert aber erheblichen Vorabdesignaufwand und laufende Governance.

**Vorteile:**

- Bietet ein einziges maßgebliches Dokument, das alle Zugriffsrechte definiert, was Sicherheitsaudits einfach und effizient macht.
- Ermöglicht konsistente Implementierung von Zugriffskontrollen im gesamten System, indem Geschäftsanforderungen in technische Berechtigungen übersetzt werden.
- Unterstützt Compliance mit Vorschriften, die nachweisbare Zugriffskontrollrichtlinien erfordern (GDPR, HIPAA, SOX, PCI DSS).
- Verringert das Risiko der Berechtigungsanhäufung, indem formale Prozesse für Zugriffsgewährung und -überprüfung etabliert werden.

**Kosten und Risiken:**

- Die Erstellung des anfänglichen Autorisierungskonzepts für ein Legacy-System mit Jahren von Ad-hoc-Berechtigungen erfordert erheblichen Analyseaufwand, um bestehende Zugriffsmuster zu verstehen.
- Die Durchsetzung des Konzepts könnte die Entfernung von Berechtigungen erfordern, an die sich Nutzer gewöhnt haben, was Widerstand und mögliche Workflow-Störungen verursacht.
- Das Autorisierungskonzept muss gepflegt werden, während sich System und Organisation weiterentwickeln; ein veraltetes Konzept bietet falsche Sicherheit.
- Übermäßig komplexe Rollenhierarchien können schwierig zu verstehen und zu verwalten werden, was neue Risiken durch Fehlkonfiguration schafft.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie ein Autorisierungskonzept Ordnung in die Zugriffskontrolle in Legacy-Systemen bringt.

Ein Legacy-Versicherungsschadensystem ist seit 12 Jahren in Produktion. In dieser Zeit wurden Berechtigungen anfragebasiert ohne übergreifendes Modell gewährt. Ein Audit offenbart, dass 40 % der Nutzer administrativen Zugriff haben, den sie nicht benötigen, einschließlich der Fähigkeit, Schadensbeträge zu ändern und Zahlungen zu genehmigen. Das Team erstellt ein umfassendes Autorisierungskonzept, das fünf Kernrollen definiert (Schadenssachbearbeiter, Schadensregulierer, Schadensmanager, Auditor, Systemadministrator) mit klar dokumentierten Berechtigungen für jede. Aufgabentrennungsregeln verhindern, dass Regulierer ihre eigenen Schäden genehmigen. Das Team kartiert alle 800 Nutzer auf die passenden Rollen und entfernt unnötigen administrativen Zugriff von 320 Konten. Das Autorisierungskonzept wird als lebendes Dokument mit vierteljährlichem Überprüfungszyklus formalisiert, und die Rollenstruktur des Schadensystems wird refaktoriert, um dem Konzept genau zu entsprechen.

Ein Legacy-Regierungsleistungssystem muss neuen Datenschutzvorschriften entsprechen, die nachweisbare Zugriffskontrollen für Bürgerdaten erfordern. Das System hat kein dokumentiertes Autorisierungsmodell — Entwickler fügen Berechtigungsprüfungen ad hoc hinzu, wenn Regulatoren spezifische Bedenken äußern. Das Team entwickelt ein Autorisierungskonzept, das alle Bürgerdaten nach Sensibilitätsstufe klassifiziert, Rollen für jede Abteilung definiert, die mit dem System interagiert, und spezifiziert, auf welche Datenfelder jede Rolle zugreifen darf. Das Konzept beinhaltet Datenmaskierungsregeln (z. B. werden Sozialversicherungsnummern als XXX-XX-1234 für alle Rollen außer autorisierten Sachbearbeitern angezeigt). Die Implementierung verringert die Anzahl der Nutzer mit Zugriff auf unmaskierte sensible Daten von 200 auf 35, und das dokumentierte Konzept besteht die regulatorische Prüfung im ersten Versuch.
