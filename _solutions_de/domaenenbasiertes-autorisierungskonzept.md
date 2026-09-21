---
title: Domänenbasiertes Autorisierungskonzept
description: Steuerung des Zugriffs auf sensible Daten basierend auf fachlichen
  Berechtigungen.
category:
- Security
- Architecture
problems:
- authorization-flaws
- data-protection-risk
- regulatory-compliance-drift
- secret-management-problems
- poor-domain-model
- authorization-role-explosion
layout: solution
lang: de
en_slug: domain-based-authorization-concept
related_solutions:
- slug: authorization-concept
  similarity: 0.85
- slug: authorization
  similarity: 0.8
- slug: role-based-access-control
  similarity: 0.75
- slug: least-privilege
  similarity: 0.65
- slug: role-model-rationalization
  similarity: 0.6
- slug: domain-modeling
  similarity: 0.6
---

## Description

Ein domänenbasiertes Autorisierungskonzept definiert Zugriffskontrollregeln in Begriffen von Geschäftsrollen, Verantwortlichkeiten und Dateneigentümerschaft — wer behandelt diesen Patienten, wem gehört diese Bestellung — statt in Begriffen niedrigstufiger technischer Berechtigungen, die direkt an Datenbanktabellen, Spalten oder Systemressourcen angehängt sind. Diese Neuformulierung ist wichtig, weil Legacy-Systeme ihre Berechtigungsmodelle häufig über viele Jahre opportunistisch entwickelt haben, wobei Zugriff auf welcher technischen Ebene auch immer gerade bequem war gewährt wurde, was genau die Art angehäufter, nicht auditierbarer Überberechtigung erzeugt, die niemand nach genug verstrichener Zeit vollständig erklären kann. Autorisierung stattdessen in Geschäftsbegriffen auszudrücken bedeutet, dass jede Regel direkt gegen eine tatsächliche Geschäftsrichtlinie validiert werden kann, von jemandem, der diese Richtlinie versteht, statt einen technischen Übersetzungsschritt zu erfordern, der sowohl Fehler als auch Mehrdeutigkeit einführt. Diese Logik zu zentralisieren — statt Berechtigungsprüfungen über die Legacy-Codebasis zu verstreuen, wo auch immer ein Entwickler einst entschied, dass eine Prüfung nötig war — macht die resultierenden Regeln auch als einzelnes Artefakt auditierbar, was für regulatorische Compliance in Domänen wie Gesundheitswesen oder Finanzen essenziell ist. Dieses Modell nachträglich auf ein Legacy-System anzuwenden erfordert zunächst, die bestehenden, oft undokumentierten Zugriffsmuster des Systems gegen das abzubilden, was das Geschäft tatsächlich beabsichtigt, ein Prozess, der zuverlässig Jahre exzessiver Berechtigungen zutage fördert, die durch Ad-hoc-Anfragen gewährt wurden, die nie überprüft oder widerrufen wurden.

## How to Apply ◆

- Definieren Sie Autorisierungsregeln in Begriffen von Geschäftsrollen und Dateneigentümerschaft statt technischer Berechtigungen auf Systemressourcen.
- Bilden Sie das aktuelle Zugriffskontrollmodell des Legacy-Systems gegen tatsächliche Geschäftsautorisierungsanforderungen ab, um Lücken und Überberechtigungen zu identifizieren.
- Implementieren Sie attributbasierte Zugriffskontrolle (ABAC) oder rollenbasierte Zugriffskontrolle (RBAC), ausgerichtet an Geschäftsdomänenkonzepten.
- Zentralisieren Sie Autorisierungslogik, statt Berechtigungsprüfungen über die Legacy-Codebasis zu verstreuen.
- Prüfen Sie bestehende Zugriffsmuster, um Nutzer mit exzessiven, über Jahre durch Ad-hoc-Gewährungen angehäuften Berechtigungen zu entdecken.
- Testen Sie Autorisierungsregeln gegen Geschäftsszenarien, um sicherzustellen, dass sensible Daten gemäß regulatorischer Anforderungen geschützt sind.

## Tradeoffs ⇄

**Vorteile:**
- Autorisierungsregeln spiegeln tatsächliche Geschäftsrichtlinien wider, was sie für Geschäfts-Stakeholder leichter validierbar macht.
- Reduziert das Risiko unautorisierten Datenzugriffs, indem Berechtigungen an Geschäftsabsicht ausgerichtet werden.
- Unterstützt regulatorische Compliance, indem auditierbare, geschäftlich bedeutungsvolle Zugriffskontrollen bereitgestellt werden.

**Kosten:**
- Domänenbasierte Autorisierung nachträglich in ein Legacy-System mit Ad-hoc-Zugriffskontrollen einzubauen ist komplex.
- Erfordert tiefes Verständnis sowohl der Geschäftsdomäne als auch des aktuellen Berechtigungsmodells des Legacy-Systems.
- Übermäßig restriktive Autorisierung kann legitime Arbeitsabläufe behindern, wenn Geschäftsrollen zu eng definiert sind.
- Zentralisierte Autorisierung wird zu einer kritischen Komponente, die hochverfügbar sein muss.

## How It Could Be

Ein Legacy-Krankenhausinformationssystem gewährt Nutzern Berechtigungen auf Datenbankebene, was dazu führt, dass Pflegekräfte Zugriff auf Abrechnungsdaten haben und Verwaltungspersonal klinische Aufzeichnungen sieht. Über die Jahre haben sich Berechtigungen ohne Überprüfung angehäuft, und niemand ist sich sicher, wer worauf Zugriff hat. Das Team führt ein domänenbasiertes Autorisierungsmodell ein, bei dem der Zugriff durch klinische Rolle (Arzt, Pflegekraft, Apotheker) und Patientenbeziehung (Behandlungsteam, konsultierend, keine Beziehung) gesteuert wird. Autorisierungsregeln werden in Geschäftsbegriffen ausgedrückt: „Pflegekräfte im Behandlungsteam des Patienten können Vitalwerte und Medikamentenanordnungen sehen, aber keine Abrechnungsinformationen." Die verstreuten Berechtigungsprüfungen des Legacy-Systems werden zu einem Autorisierungsdienst konsolidiert. Eine umfassende Prüfung deckt Hunderte exzessiver Berechtigungen auf und widerruft sie, was die Compliance-Position des Krankenhauses erheblich verbessert.
