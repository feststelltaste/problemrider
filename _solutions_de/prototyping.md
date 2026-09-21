---
title: Prototyping
description: Frühes Einholen von Feedback zu Funktionalität und Nutzbarkeit.
category:
- Process
- Requirements
problems:
- assumption-based-development
- implementation-rework
- requirements-ambiguity
- poor-user-experience-ux-design
- misaligned-deliverables
- fear-of-change
- difficulty-quantifying-benefits
- rapid-prototyping-becoming-production
layout: solution
lang: de
en_slug: prototyping
related_solutions:
- slug: prototypes
  similarity: 0.95
- slug: wireframing
  similarity: 0.8
- slug: on-site-customer
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
- slug: functional-spike
  similarity: 0.75
---

## Description

Prototyping ist die Praxis, gering verpflichtende Darstellungen einer vorgeschlagenen Änderung zu bauen — von Papierskizzen bis klickbaren Mockups bis eng begrenzt funktionierendem Code —, speziell um Unsicherheit darüber zu reduzieren, ob Nutzer, die an einen bestehenden Legacy-Workflow gewöhnt sind, einen vorgeschlagenen Ersatz akzeptieren werden, bevor dieser Ersatz vollständig gebaut wird. Anders als ein fertiggestelltes Inkrement des Systems liegt der Wert eines Prototyps vollständig im Feedback, das er erzeugt: Die Genauigkeit wird bewusst für die gestellte Frage gewählt, und eine explizite Vereinbarung, dass der Code des Prototyps verworfen und neu geschrieben wird, wird als Teil der Praxis behandelt, nicht als nachträglicher Gedanke. Speziell in der Legacy-Modernisierung wird Prototyping genutzt, um die zwei schwierigsten Quellen von Unsicherheit in einem Ersatzaufwand zu entschärfen: ob das neue Design zu Arbeitsabläufen passt, die Nutzer nie artikulieren mussten, weil das Legacy-System es einfach so macht, und ob ein vorgeschlagener Integrationsansatz gegen eine Legacy-Datenbank oder -API tatsächlich funktionieren wird, bevor Ingenieurzeit in den vollständigen Bau gebunden wird. Strukturierte Feedback-Sitzungen, in denen Legacy-Nutzer den Prototyp direkt mit ihrer aktuellen Aufgabe vergleichen, statt ihn abstrakt zu bewerten, sind das, was eine subjektive Designmeinung in konkreten, umsetzbaren Input für das Backlog verwandelt. Der wiederkehrende Fehlerfall ist, dass Prototyp-Code, gebaut unter demselben Zeitdruck wie der Rest des Projekts, still zu Produktionscode wird — eine Abkürzung, die genau die technischen Schulden wieder einführt, die die Modernisierungsanstrengung reduzieren sollte, weshalb die Etablierung der Prototyp/Produktion-Grenze im Voraus als untrennbar von der Praxis selbst behandelt wird.

## How to Apply ◆

> Prototyping in Legacy-Kontexten konzentriert sich darauf, Unsicherheit darüber zu reduzieren, ob eine vorgeschlagene Änderung oder ein Ersatz Nutzer zufriedenstellen wird, die an spezifische Legacy-Arbeitsabläufe gewöhnt sind.

- Identifizieren Sie die riskantesten Aspekte der Modernisierung — die Features, bei denen Legacy-Verhalten am wenigsten verstanden ist oder wo das Ersatzdesign am meisten abweicht — und prototypen Sie diese zuerst.
- Wählen Sie die angemessene Genauigkeitsstufe: Papierskizzen für Workflow-Validierung, klickbare Mockups für UI-Feedback oder funktionierende Code-Prototypen für technische Machbarkeit.
- Etablieren Sie eine klare „Prototyp-Grenze" mit Stakeholdern: Vereinbaren Sie im Voraus, dass Prototyp-Code verworfen und mit angemessenen Engineering-Praktiken neu geschrieben wird.
- Führen Sie strukturierte Feedback-Sitzungen durch, in denen Legacy-System-Nutzer Prototyp-Arbeitsabläufe mit ihren aktuellen Aufgaben vergleichen und festhalten, wo der Prototyp ihre Erfahrung verbessert, gleicht oder verschlechtert.
- Nutzen Sie Prototypen, um Integrationsansätze mit Legacy-Systemen zu testen — zum Beispiel das Prototypen eines API-Wrappers um eine Legacy-Datenbank, um Datenzugriffsmuster zu validieren, bevor Sie sich auf eine vollständige Implementierung festlegen.
- Verfolgen Sie Prototyp-Feedback systematisch und speisen Sie es als validierte Anforderungen in das Produkt-Backlog ein.

## Tradeoffs ⇄

> Prototyping tauscht anfängliche Zeit gegen reduzierten Nacharbeitsaufwand und verbesserte Anforderungsklarheit, erfordert aber Disziplin, um zu verhindern, dass Prototyp-Code zu Produktionscode wird.

**Vorteile:**

- Erfasst Missverständnisse bei Anforderungen und Usability-Probleme Wochen oder Monate, bevor sie in einer Produktionsimplementierung auftauchen würden.
- Hilft, die Kommunikationslücke zwischen Entwicklern, die in technischen Begriffen denken, und Nutzern, die in Arbeitsabläufen und Geschäftsergebnissen denken, zu überbrücken.
- Liefert konkrete Evidenz für Modernisierungsinvestitionsentscheidungen statt sich auf theoretische Argumente zu verlassen.
- Reduziert Widerstand gegen Veränderung, indem Nutzer Verbesserungen aus erster Hand erleben, statt darüber erzählt zu bekommen.

**Kosten und Risiken:**

- In Produktion durchsickernder Prototyp-Code ist eine häufige Quelle technischer Schulden in Modernisierungsprojekten, besonders wenn Teams unter Zeitdruck stehen.
- Prototyping ohne klare Ziele kann zu offener Erkundung entarten, die die tatsächliche Entwicklung verzögert.
- Nutzer könnten starke Bindungen an spezifische Prototyp-Designs entwickeln, was es schwierig macht, Feedback von anderen Nutzergruppen einzubeziehen.
- Der für Prototyping benötigte Aufwand könnte von Stakeholdern, die linearen Fortschritt bis zur Auslieferung erwarten, als verschwenderisch angesehen werden.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Prototyping Entscheidungsfindung in der Legacy-Modernisierung leitet.

Ein Fertigungsunternehmen musste sein Werkstatt-Planungssystem modernisieren, aber die Bediener waren zutiefst skeptisch, dass irgendein Ersatz die komplexe, restriktionsbasierte Planung handhaben könnte, die sie täglich durchführten. Das Team baute einen funktionierenden Prototyp, der eine vereinfachte Version des Planungsproblems handhabte, und lud drei erfahrene Bediener ein, ihn mit echten Produktionsdaten zu testen. Die Bediener identifizierten schnell, dass die Drag-and-Drop-Oberfläche des Prototyps für routinemäßige Planungsänderungen schneller war, aber die Fähigkeit fehlte, maschinenspezifische Einschränkungen auszudrücken, die das Legacy-System über obskure Tastenkombinationen handhabte. Dieses Feedback führte zu einem hybriden Oberflächendesign, das moderne UI-Muster mit einem Einschränkungs-Ausdrucks-Panel kombinierte und sowohl Usability-Ziele als auch Power-User-Anforderungen erfüllte. Die Prototyp-Sitzungen verwandelten auch den skeptischsten Bediener in einen Fürsprecher der Modernisierungsanstrengung.
