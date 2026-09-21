---
title: Formular-Design und mehrstufige Assistenten
description: Strukturierung komplexer Dateneingabe durch gruppierte Felder, Assistenten
  und bedingte Sichtbarkeit.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/form-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- increased-error-rates
- cognitive-overload
- negative-user-feedback
- increased-cognitive-load
- customer-dissatisfaction
layout: solution
lang: de
en_slug: form-design
related_solutions:
- slug: progressive-disclosure
  similarity: 0.75
- slug: visual-hierarchy
  similarity: 0.75
- slug: real-time-input-validation
  similarity: 0.75
- slug: cognitive-load-minimization
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
- slug: responsive-design
  similarity: 0.7
---

## Description

Gutes Formular-Design gruppiert zusammengehörige Felder, teilt lange Formulare in fokussierte mehrstufige Assistenten auf und zeigt Felder nur, wenn sie relevant sind, statt jedes Feld, das eine Legacy-Datenbanktabelle zufällig hat, auf einem überwältigenden Bildschirm zu präsentieren. Weil Legacy-Formulare typischerweise in der Reihenfolge angeordnet wurden, die das Schema vorgab, statt in der Reihenfolge, in der ein Nutzer eine Aufgabe tatsächlich durchdenkt, zwingen sie Nutzer, die gesamte Formularstruktur im Kopf zu behalten, nur um die Handvoll Felder zu finden, die für ihren spezifischen Fall relevant sind. Die Umstrukturierung um den tatsächlichen Workflow des Nutzers herum — mit Inline-Validierung pro Schritt statt einer Fehlerwand bei der abschließenden Absendung — verringert sowohl Fehlerraten als auch Abbruchquoten, auf Kosten sorgfältigen State-Managements, um Daten über Schritte hinweg zu bewahren, in einem oft nie dafür gebauten Legacy-Frontend.

## How to Apply ◆

> Legacy-Systeme präsentieren häufig alle Dateneingabefelder auf einem einzigen Bildschirm in der Reihenfolge, in der sie in der Datenbank erscheinen, was Nutzer mit Dutzenden Feldern auf einmal überwältigt. Durchdachtes Formular-Design verringert Fehler und verbessert Abschlussquoten.

- Gruppieren Sie zusammengehörige Felder visuell mit Fieldsets, Überschriften und Weißraum. Felder, die zum selben logischen Konzept gehören, wie Adressfelder oder Kontaktinformationen, sollten zusammen erscheinen.
- Teilen Sie lange Formulare in mehrstufige Assistenten mit klaren Fortschrittsanzeigen auf, die zeigen, wie viele Schritte verbleiben und was jeder Schritt abdeckt. Jeder Schritt sollte sich auf eine logische Informationsgruppe fokussieren.
- Implementieren Sie bedingte Feldsichtbarkeit, sodass Felder nur erscheinen, wenn sie basierend auf vorherigen Auswahlen relevant sind. Legacy-Formulare, die jedes mögliche Feld unabhängig vom Kontext zeigen, verschwenden Nutzeraufmerksamkeit.
- Bieten Sie Inline-Validierung nach jedem Feld oder Schritt, statt bis zur abschließenden Absendung zu warten, um alle Fehler auf einmal zu melden. Zeigen Sie spezifische, handlungsleitende Meldungen neben dem zu korrigierenden Feld.
- Setzen Sie sinnvolle Standardwerte für Felder, wo eine übliche Wahl existiert. Befüllen Sie automatisch Felder, die aus zuvor eingegebenen Daten oder dem Nutzerprofil abgeleitet werden können.
- Fügen Sie Zusammenfassungsbildschirme vor der abschließenden Absendung hinzu, die Nutzern erlauben, alle eingegebenen Daten zu überprüfen und zu bestimmten Schritten zurückzugehen, um zu korrigieren, ohne ihren Fortschritt zu verlieren.

## Tradeoffs ⇄

> Gut gestaltete Formulare verringern Fehlerraten und Nutzerfrustration dramatisch, aber die Umstrukturierung von Legacy-Formularen erfordert Verständnis sowohl des Datenmodells als auch des Nutzer-Workflows.

**Vorteile:**

- Verringert Formularabbrüche und Fehlerraten, indem Information in handhabbaren Abschnitten präsentiert wird statt in überwältigenden Feldmauern.
- Verringert kognitive Überlastung, indem nur die für die aktuellen Auswahlen des Nutzers relevanten Felder gezeigt werden, was Komplexität verbirgt, die sie nicht verwalten müssen.
- Verbessert die Datenqualität, weil Inline-Validierung Fehler früh fängt und bedingte Felder Nutzer davon abhalten, irrelevante Information einzugeben.
- Verkürzt die gefühlte Formularausfüllzeit, selbst wenn dieselbe Zahl an Feldern erfasst wird, weil Fortschrittsanzeigen klare Erwartungen setzen.

**Kosten und Risiken:**

- Ein einseitiges Formular in einen Assistenten aufzuteilen erfordert sorgfältiges State-Management, um Daten über Schritte hinweg zu bewahren, was in Legacy-Frontend-Architekturen komplex sein kann.
- Bedingte Feldlogik kann schwer zu pflegen werden, während sich Geschäftsregeln weiterentwickeln, was eine zweite Komplexitätsschicht über der Backend-Validierung schafft.
- Mehrstufige Formulare verbergen den Gesamtumfang der erforderlichen Information, was Nutzer frustrieren kann, die alles auf einmal sehen und Felder in ihrer bevorzugten Reihenfolge ausfüllen möchten.
- Die Konvertierung bestehender Formularlayouts erfordert Koordination mit Backend-Validierungslogik, die möglicherweise erwartet, dass alle Felder gleichzeitig übermittelt werden.

## How It Could Be

> Legacy-Dateneingabeformulare sind oft der schmerzhafteste Teil der Nutzererfahrung, weil sie für die Datenbank statt für den Nutzer entworfen wurden.

Ein Legacy-Versicherungsschadensystem verlangt von Sachbearbeitern, ein einziges Formular mit über fünfzig Feldern auszufüllen, um einen neuen Schaden zu melden. Das Formular enthält Felder für jeden möglichen Schadentyp, einschließlich Fahrzeugschaden, Sachschaden, Personenschaden und Haftung, alle gleichzeitig sichtbar, unabhängig vom tatsächlichen Schadentyp. Sachbearbeiter reichen regelmäßig unvollständige oder falsche Schäden ein, weil sie den Überblick verlieren, welche Felder für ihren Fall gelten. Das Team strukturiert das Formular in einen fünfstufigen Assistenten um: Schadentyp-Auswahl, Versicherungsnehmer-Information, Vorfalldetails (bedingt gezeigt basierend auf Schadentyp), Dokumenten-Upload und Zusammenfassung/Überprüfung. Jeder Schritt validiert seine Felder, bevor der Fortschritt zum nächsten erlaubt wird. Schadenmeldungsfehler sinken um über vierzig Prozent, und die durchschnittliche Zeit zur Schadenmeldung sinkt, weil Sachbearbeiter keine Zeit mehr damit verbringen, herauszufinden, welche Felder sie überspringen sollen.
