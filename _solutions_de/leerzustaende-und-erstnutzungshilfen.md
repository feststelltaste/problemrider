---
title: Leerzustände und Erstnutzungshilfen
description: Gestaltung aussagekräftiger Leerzustände mit klarer Anleitung, was als
  Nächstes zu tun ist.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/empty-states-and-first-use-guidance/
problems:
- user-confusion
- poor-user-experience-ux-design
- user-frustration
- difficult-developer-onboarding
- negative-user-feedback
- inadequate-onboarding
- feature-gaps
layout: solution
lang: de
en_slug: empty-states-and-first-use-guidance
related_solutions:
- slug: integrated-onboarding
  similarity: 0.75
- slug: intuitive-navigation
  similarity: 0.7
- slug: visual-hierarchy
  similarity: 0.7
- slug: contextual-help
  similarity: 0.7
- slug: understandable-error-messages
  similarity: 0.7
- slug: plain-language
  similarity: 0.7
---

## Description

Ein Leerzustand ersetzt einen leeren Bildschirm — den Standard, den ein Legacy-System zeigt, wenn schlicht noch keine Daten vorhanden sind — durch eine klare Erklärung, warum nichts da ist und was als Nächstes zu tun ist, sodass ein neuer Nutzer nicht raten muss, ob das System kaputt ist oder ihm etwas fehlt. Legacy-Oberflächen sagen in dieser Situation routinemäßig überhaupt nichts, was ein besonders schlechter erster Eindruck für jeden ist, der dem System zum ersten Mal begegnet, da es genau in dem Moment, in dem Anleitung am wichtigsten wäre, keinen Weg nach vorn bietet. Diese Zustände mit einem konkreten Handlungsaufruf zu gestalten und sie visuell von echten Fehlerzuständen zu unterscheiden verwandelt eine sonst verwirrende Sackgasse in das erste Stück Onboarding des Systems.

## How to Apply ◆

> Legacy-Systeme zeigen oft leere Bildschirme oder leere Tabellen ohne Erklärung an, wenn keine Daten zum Anzeigen vorhanden sind, was Nutzer verwirrt zurücklässt, ob das System kaputt ist oder sie handeln müssen.

- Identifizieren Sie alle Bildschirme im Legacy-System, die leer erscheinen können, einschließlich Listen, Dashboards, Suchergebnisse und Detailansichten für neue Konten oder Projekte.
- Ersetzen Sie leere Bildschirme durch informative Leerzustände, die erklären, warum keine Daten vorhanden sind und was der Nutzer als Nächstes tun kann. Beziehen Sie einen klaren Handlungsaufruf ein wie „Erstellen Sie Ihr erstes Projekt" oder „Importieren Sie Daten, um zu beginnen".
- Gestalten Sie Erstnutzungserfahrungen für neue Nutzer oder neue Feature-Bereiche, die Nutzer durch die anfänglichen Einrichtungsschritte führen, statt sie in eine leere Oberfläche fallen zu lassen.
- Nutzen Sie Illustrationen oder Icons in Leerzuständen, um sie visuell von Fehlerzuständen zu unterscheiden. Nutzer sollten sofort verstehen, dass das Fehlen von Daten erwartet ist, keine Fehlfunktion.
- Bieten Sie Beispiel- oder Demo-Daten an, die neue Nutzer erkunden können, um das System zu verstehen, bevor sie ihre eigenen Daten einbringen. Dies ist besonders wertvoll in komplexen Legacy-Systemen mit steilen Lernkurven.
- Testen Sie Leerzustände mit tatsächlichen neuen Nutzern, um zu verifizieren, dass die Anleitung ausreichend und die Handlungsaufrufe klar sind.

## Tradeoffs ⇄

> Aussagekräftige Leerzustände verwandeln potenziell verwirrende Momente in Gelegenheiten für Nutzerbildung, erfordern aber Design- und Inhaltsaufwand.

**Vorteile:**

- Eliminiert Nutzerverwirrung beim Auftreffen auf Bildschirme ohne Daten, was in Legacy-Systemen eine häufige Quelle von Support-Anfragen ist.
- Verbessert das Onboarding, indem neue Nutzer durch ihre ersten Aktionen geleitet werden, statt sie das System selbst herausfinden zu lassen.
- Reduziert die wahrgenommene Komplexität des Systems, indem klare Einstiegspunkte für den Start geboten werden.
- Verhindert, dass Nutzer annehmen, das System sei kaputt oder ihnen fehle Zugriff, wenn sie leere Bildschirme sehen.

**Kosten und Risiken:**

- Das Gestalten und Implementieren von Leerzuständen für jeden möglichen leeren Bildschirm in einem großen Legacy-System erfordert Inhaltsschreibung und Designaufwand.
- Leerzustandsinhalte müssen gepflegt werden, während sich das System weiterentwickelt; veraltete Anleitung, die auf entfernte Features verweist, ist verwirrend.
- Erstnutzungsanleitung kann für Nutzer lästig werden, die häufig neue Projekte oder Konten erstellen, was einen Mechanismus zum Überspringen oder Ausblenden erfordert.
- Die Lokalisierung von Leerzustandsinhalten in mehrere Sprachen erhöht die Übersetzungslast.

## How It Could Be

> Der erste Eindruck eines Legacy-Systems für neue Nutzer ist oft ein Bildschirm ohne Daten und ohne Anleitung, was von Anfang an einen negativen Ton setzt.

Ein Legacy-Projektmanagementsystem präsentiert neuen Nutzern nach ihrem ersten Login ein vollständig leeres Dashboard. Die Seitenleiste enthält kryptische Menübezeichnungen wie „WBS", „CR Log" und „Baseline Control". Neue Projektmanager, die mit der Terminologie des Systems nicht vertraut sind, eröffnen Support-Tickets mit der Frage, wie sie das System nutzen sollen. Das Team gestaltet das leere Dashboard neu, um eine Willkommensnachricht mit drei klaren Schritten anzuzeigen: „Erstellen Sie Ihr erstes Projekt", „Laden Sie Teammitglieder ein" und „Richten Sie Ihren ersten Meilenstein ein". Jeder Schritt verlinkt direkt auf den relevanten Bildschirm mit einer kurzen Erklärung, was er tut. Das Team fügt auch hilfreiche Leerzustände zu den Bildschirmen für Projektliste, Aufgabenboard und Dokumentenrepository hinzu. Neue-Nutzer-Support-Tickets bezüglich der Ersteinrichtung sinken erheblich, und die durchschnittliche Zeit von der Kontoerstellung bis zur ersten bedeutsamen Aktion sinkt von Tagen auf Stunden.
